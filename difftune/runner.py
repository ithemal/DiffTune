#!/usr/bin/env python3

import argparse
import cloudpickle as pickle
import collections
import functools
import itertools
import numpy as np
import operator
import os
import pandas as pd
import multiprocessing.dummy as mp
import random
import re
import subprocess
import tempfile
import time
import torch
import tqdm
import xml.etree.ElementTree as ET
import numba
import shutil
from . import model

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_PARENT_DIRNAME = os.path.dirname(_DIRNAME)

LLVM_BIN_DIR = os.environ.get('LLVM_BIN_DIR', os.path.join(_PARENT_DIRNAME, 'llvm-mca-parametric', 'build', 'bin'))
LLVM_SIM_DIR = os.environ.get('LLVM_SIM_DIR', os.path.join(_PARENT_DIRNAME, 'exegesis-parametric'))
DATA_BASE = os.environ.get('DATA_BASE', 'data')
TMPDIR = os.environ.get('TMPDIR', '/tmp')

LLVM_MC = os.path.join(LLVM_BIN_DIR, 'llvm-mc')
LLVM_MCA = os.path.join(LLVM_BIN_DIR, 'llvm-mca')
LLVM_GET_TABLES = os.path.join(LLVM_BIN_DIR, 'llvm-get-tables')
LLVM_SIM = os.path.join(LLVM_SIM_DIR, 'bazel-bin', 'llvm_sim', 'x86', 'faucon')
GITROOT = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

is_child = False

def reseed_numpy():
    global is_child
    np.random.seed(os.getpid())
    is_child = True

os.register_at_fork(
    after_in_child=reseed_numpy,
)

### CODE AND PORT MAPS ###

def get_port_map_for_arch(arch):
    proc = subprocess.Popen(
        [LLVM_MCA, '-parameters', 'noop',
         '-mtriple=x86_64-unknown-unknown', '-march=x86-64', '-mcpu={}'.format(arch),
         '--all-views=0', '-iterations=1', '-debug'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        universal_newlines=True
    )
    (_, err) =  proc.communicate('popq %rax')
    err = err.split('\n')
    masks_idx = err.index('Processor resource masks:') + 1
    masks = {}
    while err[masks_idx].strip():
        _, name, _, mask = err[masks_idx].split()
        masks[int(mask)] = name
        masks_idx += 1
    return masks

### PARAMETER TABLE

SampleTableParameters = collections.namedtuple('SampleTableParameters', [
    'seed', 'min_latency', 'max_latency', 'min_n_uops', 'max_n_uops', 'min_n_ports', 'max_n_ports', 'min_port_time', 'max_port_time', 'by_schedclass', 'use_groups',
    'min_dispatch_width', 'max_dispatch_width', 'min_microop_buffer_size', 'max_microop_buffer_size', 'min_readadvance', 'max_readadvance',
    'min_microops', 'max_microops',
])

SampleTableResult = collections.namedtuple('SampleTableResult', [
    'instruction_table', 'global_vector',
])

def get_sample_table_parameters(
        seed,
        min_latency=0,
        max_latency=10,
        min_n_uops=1,
        max_n_uops=10,
        min_n_ports=0,
        max_n_ports=5,
        min_port_time=1,
        max_port_time=5,
        min_dispatch_width=1,
        max_dispatch_width=10,
        min_microop_buffer_size=50,
        max_microop_buffer_size=250,
        min_readadvance=0,
        max_readadvance=10,
        min_microops=0,
        max_microops=10,
        by_schedclass=False,
        use_groups=True,
):
    return SampleTableParameters(
        seed=seed,
        min_latency=min_latency,
        max_latency=max_latency,
        min_n_uops=min_n_uops,
        max_n_uops=max_n_uops,
        min_n_ports=min_n_ports,
        max_n_ports=max_n_ports,
        min_port_time=min_port_time,
        max_port_time=max_port_time,
        min_dispatch_width=min_dispatch_width,
        max_dispatch_width=max_dispatch_width,
        min_microop_buffer_size=min_microop_buffer_size,
        max_microop_buffer_size=max_microop_buffer_size,
        min_readadvance=min_readadvance,
        max_readadvance=max_readadvance,
        min_microops=min_microops,
        max_microops=max_microops,
        by_schedclass=by_schedclass,
        use_groups=use_groups,
    )

N_READADVANCE_PARAMS = 7

@numba.njit
def do_sample(table,
              lat_col, latency_samples,
              n_ports_samples, portcol_samples, portval_samples,
              readadvance_offset, readadvance_sample,
              microops_col, microops_sample,
):
    for i in range(len(table)):
        table[i, lat_col] = latency_samples[i]
        table[i, microops_col] = microops_sample[i]
        for j in range(n_ports_samples[i]):
            table[i, portcol_samples[i, j]] = portval_samples[i, j]
        for j in range(7):
            table[i, readadvance_offset + j] = readadvance_sample[i, j]


group_port_re = re.compile('^(?P<name>HW|SKL|SB)(?P<typ>Divider|FPDivider|Port(?P<group>Any|\d+))$')
single_port_re = re.compile('^(?P<name>HW|SKL|SB)(?P<typ>Divider|FPDivider|Port\d)$')

def sample_table(
        from_, params, opcodes=None, shuffle=False
):
    if shuffle:
        raise NotImplementedError()
        tab = from_.sample(frac=1, random_state=params.seed)
        tab.index = from_.index
        return tab.instruction_table.reindex(opcodes)

    tab = from_.instruction_table.reindex(opcodes, fill_value=0)
    n_rows = len(tab)

    if params.use_groups:
        port_cols = [i for (i, c) in enumerate(tab.columns) if group_port_re.match(c)]
    else:
        port_cols = [i for (i, c) in enumerate(tab.columns) if single_port_re.match(c)]

    lat_col = next(i for (i, c) in enumerate(tab.columns) if c == 'Latency')
    microops_col = next(i for (i, c) in enumerate(tab.columns) if c == 'NumMicroOps')
    rng = np.random.RandomState(params.seed)

    latency_samples = rng.randint(params.min_latency, 1+params.max_latency, n_rows)
    microops_samples = rng.randint(params.min_microops, 1+params.max_microops, n_rows)
    n_ports_samples = rng.randint(params.min_n_ports, params.max_n_ports+1, n_rows)
    portcol_samples = rng.choice(port_cols, (n_rows, params.max_n_ports), replace=True)
    portval_samples = rng.randint(params.min_port_time, params.max_port_time+1, (n_rows, params.max_n_ports))
    readadvance_samples = rng.randint(params.min_readadvance, params.max_readadvance+1, (n_rows, N_READADVANCE_PARAMS))
    readadvance_offset = next(i for (i, c) in enumerate(tab.columns) if 'readadvance' in c.lower())

    dispatch_width = rng.randint(params.min_dispatch_width, params.max_dispatch_width + 1)
    microop_buffer_size = rng.randint(params.min_microop_buffer_size, params.max_microop_buffer_size + 1)

    table_values = np.zeros_like(tab.values)
    do_sample(table_values, lat_col, latency_samples, n_ports_samples, portcol_samples, portval_samples, readadvance_offset, readadvance_samples, microops_col, microops_samples)
    table = pd.DataFrame(
        data=table_values,
        index=tab.index,
        columns=tab.columns,
    )

    return SampleTableResult(
        instruction_table=table,
        global_vector=[dispatch_width, microop_buffer_size],
    )

def write_params_to_file(parameters, dirname=None, env=False):
    if parameters is None:
        return (False, 'noop')
    elif isinstance(parameters, str):
        return (False, parameters)

    instr_table = parameters.instruction_table
    col_indir = {n: i for (i, n) in enumerate(instr_table.columns)}
    res = [
        'dispatch-width {}'.format(parameters.global_vector[0]),
        'microop-buffer-size {}'.format(parameters.global_vector[1]),
    ]

    for opcode, param_row in zip(instr_table.index, instr_table.values):
        all_port_times = 0
        for port_name in col_indir.keys():
            if not any(p in port_name for p in ('Port', 'Divider')):
                continue
            port_time = param_row[col_indir[port_name]]
            all_port_times += port_time
            res.append('port-{}-{} {}'.format(
                opcode,
                port_name,
                port_time,
            ) )
        res.append('latency-{}-0 {}'.format(opcode, param_row[col_indir['Latency']]))
        res.append('microops-{} {}'.format(opcode, max(1, param_row[col_indir['NumMicroOps']])))
        for j in range(N_READADVANCE_PARAMS):
            res.append('readadvance-{}-{}-0 {}'.format(opcode, j, param_row[col_indir['readadvance-{}-0'.format(j)]]))


    if env:
        return (False, {v.split()[0].replace('-', '_'): v.split()[1] for v in res})
    else:
        (fd, fname) = tempfile.mkstemp(dir=dirname, text=True)
        tf = os.fdopen(fd, 'w')
        tf.write('\n'.join(res))
        tf.flush()
        tf.close()
        return (True, fname)


### ARCHES

long_arches = ['haswell', 'skylake', 'ivybridge', 'znver1']
short_arches = ['hsw', 'skl', 'ivb', 'amd']
long_to_short = {l: s for (l, s) in zip(long_arches, short_arches)}
short_to_long = {s: l for (l, s) in zip(long_arches, short_arches)}


### BLOCKS

def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        print(len(covered) / len(elements), end='\r')
        i, subset = max(enumerate(subsets), key=lambda s: len(s[1] - covered))
        cover.append(i)
        covered |= subset
    return cover

def get_default_params():
    defaults = {}
    for arch in long_arches:
        port_map = get_port_map_for_arch(arch)

        proc = subprocess.Popen(
            [LLVM_GET_TABLES, '-mtriple=x86_64-unknown-unknown', '-march=x86-64', '-mcpu={}'.format(arch), '-builtin'],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        (output, err) = proc.communicate()

        names = []
        opcodes = []
        latencies = []
        ports = []
        microops = []

        lines = output.strip().split('\n')

        dispatch_width = int(lines[0].split()[1])
        microop_buffer_size = int(lines[1].split()[1])

        for line in tqdm.tqdm(lines[2:]):
            (key, value) = line.split()
            if key == 'opcode-name':
                names.append(value)
                ports.append({})
            elif key.startswith('latency'):
                latencies.append(int(value))
                opcodes.append(int(key.split('-')[1]))
            elif key.startswith('microop'):
                microops.append(int(value))
            elif key.startswith('port'):
                ports[-1][key.split('-')[-1]] = int(value)
            elif key.startswith('readadvance'):
                key = list(key.split('-'))
                ports[-1]['readadvance-{}-{}'.format(*key[2:])] = int(value)
            else:
                raise ValueError()

        df = pd.DataFrame.from_dict(ports)
        df['OpcodeID'] = opcodes
        df['OpcodeName'] = names
        df['Latency'] = latencies
        df['NumMicroOps'] = microops

        defaults[arch] = SampleTableResult(
            instruction_table=df.set_index('OpcodeID', drop=False),
            global_vector=[dispatch_width, microop_buffer_size],
        )

    return defaults



def get_blocks():
    mca_handles = {arch: McaHandle(arch) for arch in long_arches}

    mca_dir = os.path.join(GITROOT, 'bhive')

    # read base blocks, hex, xml, and timings from csv
    blocks = pd.read_csv(
        os.path.join(mca_dir, 'all-blocks.csv'),
        sep = ',', names=['idx', 'hex'],
    ).dropna()
    for arch in short_arches:
        t = pd.read_csv(
            os.path.join(mca_dir, arch),
            sep = ',', names=['hex', '{}-true'.format(arch)],
        ).dropna()
        blocks = blocks.merge(t, left_on='hex', right_on='hex')
    t = pd.read_csv(
        os.path.join(mca_dir, 'tokens.csv'),
        sep=',', names=['hex', 'xml'],
    ).dropna()
    blocks = blocks.merge(t, left_on='hex', right_on='hex')

    t = pd.read_csv(
        os.path.join(mca_dir, 'code.csv'),
        names=['hex', 'code'],
        sep='#',
    ).dropna()
    blocks = blocks.merge(t, left_on='hex', right_on='hex')
    blocks['code'] = blocks['code'].str.replace(';', '\n')

    t = pd.read_csv(
        os.path.join(mca_dir, 'mcinsts-mca.csv'),
        names=['hex', 'mcinsts'],
    ).dropna()
    blocks = blocks.merge(t, left_on='hex', right_on='hex')

    no_aliasing = pd.read_csv(
        os.path.join(mca_dir, 'blocks-no-aliasing.csv'),
        sep = ',', names=['idx'],
    ).dropna()['idx']
    blocks = blocks[blocks['idx'].isin(no_aliasing)]
    blocks = blocks.set_index('idx')

    blocks_computed = blocks

    n_blocks = len(blocks_computed)
    all_hex_ops = list(blocks_computed[['hex', 'mcinsts']].values)
    random.seed(10)
    random.shuffle(all_hex_ops)
    all_hexes, all_opcodes = map(list, zip(*all_hex_ops))

    token_to_hot_idx = {}
    hot_idx_to_token = {}
    hexmap = {}

    def hot_idxify(elem):
        if elem not in token_to_hot_idx:
            token_to_hot_idx[elem] = len(token_to_hot_idx)
            hot_idx_to_token[token_to_hot_idx[elem]] = elem
        return token_to_hot_idx[elem]

    nmap = {
        n: i for (i, n) in enumerate(blocks_computed.columns)
    }
    for row in tqdm.tqdm(blocks_computed.values):
        code_xml = row[nmap['xml']]
        opcodes = list(map(int, row[nmap['mcinsts']].split()))

        block_root = ET.fromstring(code_xml)
        raw_instrs = []

        if len(opcodes) != len(block_root):
            continue

        for opcode, instr in zip(opcodes, block_root):
            raw_instr = []
            opcode = 'opcode-{}'.format(opcode)
            raw_instr.extend([opcode, '<SRCS>'])
            srcs = []
            for src in instr.find('srcs'):
                if src.find('mem') is not None:
                    raw_instr.append('<MEM>')
                    for mem_op in src.find('mem'):
                        raw_instr.append(int(mem_op.text))
                        srcs.append(int(mem_op.text))
                    raw_instr.append('</MEM>')
                else:
                    raw_instr.append(int(src.text))

            raw_instr.append('<DSTS>')
            dsts = []
            for dst in instr.find('dsts'):
                if dst.find('mem') is not None:
                    raw_instr.append('<MEM>')
                    for mem_op in dst.find('mem'):
                        raw_instr.append(int(mem_op.text))
                    raw_instr.append('</MEM>')
                else:
                    raw_instr.append(int(dst.text))

            raw_instr.append('<END>')
            raw_instrs.append(list(map(hot_idxify, raw_instr)))

        hexmap[row[nmap['hex']]] = raw_instrs

    blocks = blocks[blocks['hex'].isin(set(hexmap.keys()))].copy()
    blocks['tokens'] = blocks['hex'].apply(hexmap.get)
    return blocks


### MCA

class McaHandle:
    def __init__(self, arch):
        self.arch = arch
        self._port_map = get_port_map_for_arch(arch)

    def get_opcodes(self, row):
        row_code = row['code']
        proc = subprocess.Popen(
            [LLVM_MCA, '-parameters', 'noop', '-mtriple=x86_64-unknown-unknown',
             '-march=x86-64', '-mcpu=haswell', '-debug', '-iterations=1'],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        try:
            (output, err) = proc.communicate(row['code'])
        except:
            return None

        z = [x.split('Opcode= ')[1] for x in err.strip().split('\n') if 'Opcode= ' in x]
        code_name_map = {}
        for l in row_code.split('\n'):
            ident = l.split()[-1]
            if ident not in code_name_map:
                code_name_map[ident] = z[len(code_name_map)]

        return ' '.join(code_name_map[x.split()[-1]] for x in row_code.split('\n'))

    def amortize_get_timing_parameters(self, parameters):
        (_, fname) = write_params_to_file(parameters, dirname=TMPDIR)
        return functools.partial(self.get_timing, fname)

    def get_timing_fast(self, parameters, row):
        proc = subprocess.Popen(
            [LLVM_MCA, '-parameters', 'env',
             '-mtriple=x86_64-unknown-unknown', '-march=x86-64', '-mcpu={}'.format(self.arch),
             '--all-views=0', '--summary-view', '-iterations=100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
            env=parameters,
        )
        try:
            (stdout, stderr) = proc.communicate(row, timeout=10)
        except subprocess.TimeoutExpired:
            return None

        if proc.returncode:
            return None

        return int(stdout.split('\n')[2].split()[-1])

    def get_timing(self, parameters, row):
        (delete, fname) = write_params_to_file(parameters)
        proc = subprocess.Popen(
            [LLVM_MCA, '-parameters', fname,
             '-mtriple=x86_64-unknown-unknown', '-march=x86-64', '-mcpu={}'.format(self.arch),
             '--all-views=0', '--summary-view', '-iterations=100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True
        )
        try:
            (stdout, stderr) = proc.communicate(row['code'], timeout=10)
        except subprocess.TimeoutExpired:
            return None
        finally:
            if delete:
                os.unlink(fname)

        if proc.returncode:
            return stderr

        return int(stdout.split('\n')[2].split()[-1])

    def _parse_line(self, line):
        line = line.strip()
        if 'Opcode Name=' in line:
            return ('OpcodeName', line.split()[-1])
        elif 'Opcode=' in line:
            return ('OpcodeID', int(line.split()[-1]))
        elif 'SchedClassID=' in line:
            return ('SchedClassID', int(line.split('=')[-1]))
        elif 'Mask=' in line and 'Buffer' not in line:
            mask, cyc, res = map(lambda x: int(x.split('=')[1]), line.split(', '))
            return (mask, cyc)
        elif 'Def' in line and 'Latency=' in line:
            lat = int(line.strip().split(', ')[-2].split('=')[1])
            return ('Latency', lat)
        else:
            return None

class ExegesisHandle:
    def __init__(self):
        self._port_map = get_port_map_for_arch('haswell')

    def get_opcodes(self, row):
        row_code = row['code']
        proc = subprocess.Popen(
            [LLVM_SIM, '-max_iters=1', '-input_type=att_asm', '-'],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        try:
            (output, err) = proc.communicate(row['code'])
        except:
            return None

        z = [x.split('=')[1] for x in output.strip().split('\n') if 'Opcode=' in x]
        code_name_map = {}
        for l in row_code.split('\n'):
            ident = l.split('#')[0]
            if ident not in code_name_map:
                code_name_map[ident] = z[len(code_name_map)]

        return ' '.join(code_name_map[x.split('#')[0]] for x in row_code.split('\n'))

    def amortize_get_timing_parameters(self, parameters):
        (_, fname) = write_params_to_file(parameters, dirname=TMPDIR)
        return functools.partial(self.get_timing, fname)

    def get_timing(self, parameters, row):
        (delete, fname) = write_params_to_file(parameters)
        proc = subprocess.Popen(
            [LLVM_SIM, '-p', fname, '-input_type=att_asm', '-max_iters=100', '-'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True
        )
        try:
            (stdout, stderr) = proc.communicate(row['code'], timeout=10)
        except subprocess.TimeoutExpired:
            return None
        finally:
            if delete:
                os.unlink(fname)

        if proc.returncode:
            return stderr

        st = 'ran 100 iterations in'
        try:
            line = next(l for l in stdout.split('\n')[::-1] if st in l)
        except StopIteration:
            return stdout
        return int(line.split()[-2])

    def get_timings(self, parameters, rows):
        raise NotImplementedError()


def get_default_timings(blocks, defaults):

    handles = {
        arch: {
            'mca': McaHandle(arch),
        }
        for arch in long_arches
    }
    handles['haswell']['exegesis'] = ExegesisHandle()

    res = {
    }

    for arch in long_arches:
        sarch = long_to_short[arch]
        for (name, handle) in handles[arch].items():
            res['{}-{}-default'.format(sarch, name)] = blocks.apply(functools.partial(handle.get_timing, None), axis=1)

    return pd.concat(res, axis=1)


def write_sample_timings(name, sim, arch, sample_params, n_forks, shuffle):
    if sim == 'mca':
        handle = McaHandle(arch)
    elif sim == 'exegesis':
        handle = ExegesisHandle()

    try:
        assert sample_params == read(name, 'sample-params')
    except FileNotFoundError:
        write(name, 'sample-params', sample_params)

    blocks = read(name, 'blocks').reset_index()
    defaults = read(name, 'default_params')[arch]

    all_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if group_port_re.match(c)]
    m_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if (group_port_re if sample_params['use_groups'] else single_port_re).match(c)]
    readadvance_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if 'readadvance' in c]
    defaults = SampleTableResult(
        instruction_table=defaults.instruction_table[all_port_cols + readadvance_cols + ['NumMicroOps', 'Latency']],
        global_vector=defaults.global_vector,
    )

    write_f = open('{}/{}/{}-{}.csv'.format(DATA_BASE, name, sim, arch), 'a+', buffering=1)

    bmap = {k:i for (i, k) in enumerate(blocks.columns)}

    for i in range(n_forks - 1):
        if os.fork() == 0:
            break

    seed_base = np.frombuffer(np.random.bytes(4), dtype=np.uint32)[0]
    i = 0
    if not is_child:
        pbar = tqdm.tqdm()
    while True:
        i += 1
        seed = seed_base + i
        row = blocks.iloc[np.random.randint(len(blocks))]
        params = get_sample_table_parameters(seed, **sample_params)
        table = sample_table(defaults, params, sorted({int(x) for x in row['mcinsts'].split()}), shuffle=shuffle)
        timing = handle.get_timing(table, row)
        if isinstance(timing, str) or timing is None:
            continue

        write_f.write(
            '{},{},{}\n'.format(
                row['idx'],
                seed_base + i,
                timing,
            )
        )
        if not is_child:
            pbar.update(n_forks)
        if i > 3000000:
            break


class CollateResult:
    def __init__(self, instrs, instr_lens, block_lens, instr_params, global_params, timings):
        self.instrs = instrs
        self.instr_lens = instr_lens
        self.block_lens = block_lens
        self.instr_params = instr_params
        self.global_params = global_params
        self.timings = timings

    def pin_memory(self):
        self.instrs.pin_memory()
        self.instr_params.pin_memory()
        self.global_params.pin_memory()
        self.timings.pin_memory()
        return self

def do_collate(batch, defaults=None, instr_params=None, global_params=None, shuffle=False):
    assert defaults is not None

    instrs = torch.nn.utils.rnn.pad_sequence([
        torch.LongTensor(instr) for row in batch for instr in row.tokens
    ], batch_first=True)
    instr_lens = [len(instr) for block in batch for instr in block.tokens]
    block_lens = [len(block.tokens) for block in batch]
    if instr_params is None:
        tables = [
            sample_table(defaults, row.params, [int(x) for x in row.mcinsts.split()], shuffle=shuffle)
            for row in batch
        ]
        instr_params = torch.nn.utils.rnn.pad_sequence(torch.tensor([
            np.hstack((t, tbl.global_vector)) for tbl in tables for t in tbl.instruction_table.values.astype(np.float32)
        ]), batch_first=True) / 100
    else:
        instr_params = torch.nn.utils.rnn.pad_sequence([
            torch.cat((instr_params(torch.LongTensor([int(instr) - 1])).reshape(-1), global_params)).abs()
            for block in batch for instr in block.mcinsts.split(' ')
        ], batch_first=True) / 100
    global_params = torch.stack([torch.tensor([0.0]) for x in batch], dim=0)

    timings = torch.stack([torch.tensor(float(row.timing)) for row in batch], dim=0) / 10000

    return CollateResult(
        instrs,
        instr_lens,
        block_lens,
        instr_params.float(),
        global_params,
        timings,
    )

def train_approximation(name, sim, arch, model_name, model_width, model_depth, model_bidirectional, batch_size, max_delta, shuffle, opt_alg, opt_alpha, opt_beta, opt_beta_2, opt_nesterov, device, dataset_sample_size, validate, epochs):
    sample_params = read(name, 'sample-params')
    blocks = read(name, 'blocks').reset_index()

    defaults = read(name, 'default_params')[arch]
    all_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if group_port_re.match(c)]
    m_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if (group_port_re if sample_params['use_groups'] else single_port_re).match(c)]
    readadvance_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if 'readadvance' in c]
    defaults = SampleTableResult(
        instruction_table=defaults.instruction_table[all_port_cols + readadvance_cols + ['NumMicroOps', 'Latency']],
        global_vector=defaults.global_vector,
    )

    dataset = pd.read_csv('{}/{}/{}-{}.csv'.format(DATA_BASE, name, sim, arch),
                          names=['idx', 'seed', 'timing'], dtype={'idx': np.int32, 'seed': np.uint32, 'timing': np.int32})
    dataset = dataset.merge(blocks, on='idx', how='left', suffixes=('', ''))
    dataset = dataset[(dataset['timing'] - dataset['{}-true'.format(long_to_short[arch])]).abs() < max_delta]
    dataset['params'] = dataset['seed'].apply(lambda seed: get_sample_table_parameters(seed, **sample_params))

    if dataset_sample_size is not None:
        dataset = dataset.sample(dataset_sample_size)

    all_idxs = list(blocks['idx'])
    np.random.RandomState(0).shuffle(all_idxs)
    train_idxs = all_idxs[:int(0.8 * len(all_idxs))]
    val_idxs = all_idxs[int(0.8 * len(all_idxs)):int(0.9 * len(all_idxs))]
    test_idxs = all_idxs[int(0.9 * len(all_idxs)):]

    if validate:
        dataset = dataset[dataset['idx'].isin(set(val_idxs))]
    else:
        dataset = dataset[dataset['idx'].isin(set(train_idxs))]

    base_file_name = '{}-{}-{}'.format(sim, arch, model_name)

    approximation_args = (model_width, model_depth, model_bidirectional)
    args_file_name = '{}-model-args'.format(base_file_name)
    try:
        assert approximation_args == read(name, args_file_name)
    except FileNotFoundError:
        write(name, args_file_name, approximation_args)

    lstm = model.ParamLstm(
        20000,
        model_width,
        len(all_port_cols) + 11,
        1,
        model_depth,
        model_bidirectional,
    )

    model_file_name = '{}-model'.format(model_name)
    try:
        if validate:
            (n_dataitems, train_losses, lstm_state_dict) = torch.load(os.path.join(DATA_BASE, name, validate))
        else:
            (n_dataitems, train_losses, lstm_state_dict) = torch.load(os.path.join(DATA_BASE, name, '{}-latest'.format(model_file_name)))
        lstm.load_state_dict(lstm_state_dict)
        elapsed_time = train_losses[-1][1]
    except FileNotFoundError:
        n_dataitems = 0
        train_losses = []
        elapsed_time = 0

    lstm.to(device, non_blocking=True)

    start_time = time.time()

    train_dataloader = torch.utils.data.DataLoader(
        list(dataset.itertuples()), batch_size=batch_size,
        collate_fn=functools.partial(do_collate, defaults=defaults, shuffle=shuffle),
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    ema_loss = 0
    ema_idx = 0
    ema_beta = 0.995

    if opt_alg.lower() == 'adam':
        opt = torch.optim.Adam(lstm.parameters(), lr=opt_alpha,  betas=(opt_beta, opt_beta_2))
    elif opt_alg.lower() == 'sgd':
        opt = torch.optim.SGD(lstm.parameters(), lr=opt_alpha, momentum=opt_beta, nesterov=opt_nesterov)

    n_epoch = 0
    if not epochs:
        epochs = float('inf')
    while n_epoch < epochs:
        if validate:
            break
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for batch in train_dataloader:
                if (n_dataitems // batch_size) % 1000 == 0:
                    base_dir = '{}/{}'.format(DATA_BASE, name)
                    for s in ['latest', time.time()]:
                        torch.save(
                            (n_dataitems, train_losses, lstm.state_dict()),
                            os.path.join(base_dir, '{}-{}'.format(model_file_name, s)),
                        )

                preds = lstm(
                    batch.instrs.to(device, non_blocking=True),
                    batch.instr_lens,
                    batch.block_lens,
                    batch.instr_params.to(device, non_blocking=True),
                    batch.global_params.to(device, non_blocking=True),
                ).reshape(-1)
                ys = batch.timings.to(device, non_blocking=True)
                err = ((preds - ys).abs() / ys).mean()

                opt.zero_grad()
                err.backward()
                torch.nn.utils.clip_grad_norm_(lstm.parameters(), 5.0)
                opt.step()
                n_dataitems += batch_size

                ema_loss = ema_loss * ema_beta + err.item() * (1 - ema_beta)
                ema_idx += 1
                current_loss = ema_loss / (1 - ema_beta**ema_idx)

                pbar.update(batch_size)
                pbar.set_postfix(loss=current_loss)
                if len(train_losses) == 0 or n_dataitems >= train_losses[-1][0] + batch_size*100:
                    train_losses.append((n_dataitems, time.time() - start_time + elapsed_time, current_loss))
        n_epoch += 1
    if not validate:
        base_dir = '{}/{}'.format(DATA_BASE, name)
        for s in ['latest', time.time()]:
            torch.save(
                (n_dataitems, train_losses, lstm.state_dict()),
                os.path.join(base_dir, '{}-{}'.format(model_file_name, s)),
            )


    ema_loss = 0
    ema_idx = 0
    ema_beta = 0.995
    all_errs = []
    with tqdm.tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch in train_dataloader:
                preds = lstm(
                    batch.instrs.to(device, non_blocking=True),
                    batch.instr_lens,
                    batch.block_lens,
                    batch.instr_params.to(device, non_blocking=True),
                    batch.global_params.to(device, non_blocking=True),
                ).reshape(-1)
                ys = batch.timings.to(device, non_blocking=True)
                err = ((preds - ys).abs() / ys).mean()
                ema_loss = ema_loss * ema_beta + err.item() * (1 - ema_beta)
                ema_idx += 1
                current_loss = ema_loss / (1 - ema_beta**ema_idx)

                all_errs.append(err.item())

                pbar.update(batch_size)
                pbar.set_postfix(loss=current_loss)

    print(np.mean(all_errs))

def train_parameters(name, sim, arch, model_name, batch_size, shuffle, seed, params_name, opt_alg, opt_alpha, opt_beta, opt_beta_2, opt_nesterov, device, timing_arch, validate, epochs):
    version = 'latest'

    sample_params = read(name, 'sample-params')
    blocks = read(name, 'blocks').reset_index()

    defaults = read(name, 'default_params')[arch]
    all_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if group_port_re.match(c)]
    m_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if (group_port_re if sample_params['use_groups'] else single_port_re).match(c)]
    readadvance_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if 'readadvance' in c]
    defaults = SampleTableResult(
        instruction_table=defaults.instruction_table[all_port_cols + readadvance_cols + ['NumMicroOps', 'Latency']],
        global_vector=defaults.global_vector,
    )

    all_idxs = list(blocks['idx'])
    np.random.RandomState(0).shuffle(all_idxs)
    train_idxs = all_idxs[:int(0.8 * len(all_idxs))]
    val_idxs = all_idxs[int(0.8 * len(all_idxs)):int(0.9 * len(all_idxs))]
    test_idxs = all_idxs[int(0.9 * len(all_idxs)):]
    blocks = blocks[blocks['idx'].isin(set(train_idxs))]


    base_file_name = '{}-{}-{}'.format(sim, arch, model_name)

    args_file_name = '{}-model-args'.format(base_file_name)
    (model_width, model_depth, model_bidirectional) = read(name, args_file_name)

    lstm = model.ParamLstm(
        20000,
        model_width,
        len(all_port_cols) + 11,
        1,
        model_depth,
        model_bidirectional,
    )

    base_dir = '{}/{}'.format(DATA_BASE, name)
    model_file_name = '{}-model'.format(model_name)
    fname = os.path.join(base_dir, '{}-{}'.format(model_file_name, 'latest'))
    (_, train_losses, lstm_state_dict) = torch.load(fname)
    lstm.load_state_dict(lstm_state_dict)
    lstm.to(device, non_blocking=True)


    sample = sample_table(defaults, get_sample_table_parameters(seed, **sample_params), shuffle=shuffle)
    instr_params = torch.nn.Embedding(20000, len(all_port_cols) + 8)
    instr_params.weight = torch.nn.Parameter(torch.Tensor(sample.instruction_table.values.astype(np.float32)), requires_grad=True)
    global_params = torch.nn.Parameter(torch.FloatTensor(sample.global_vector), requires_grad=True)

    base_dir = '{}/{}'.format(DATA_BASE, name)
    def get_name(suffix):
        if params_name is None:
            return os.path.join(base_dir, '{}-params-{}'.format(model_file_name, suffix))
        else:
            return os.path.join(base_dir, '{}-params-{}-{}'.format(model_file_name, params_name, suffix))

    try:
        (n_dataitems, train_losses, instr_params_sd, global_params_sd) = torch.load(get_name('latest'))
        instr_params.load_state_dict(instr_params_sd)
        global_params.data.copy_(global_params_sd)
        elapsed_time = train_losses[-1][1]
    except FileNotFoundError:
        n_dataitems = 0
        elapsed_time = 0
        train_losses = []

    start_time = time.time()

    if timing_arch:
        arch = timing_arch

    blocks['timing'] = blocks['{}-true'.format(long_to_short[arch])]
    train_dataloader = torch.utils.data.DataLoader(
        list(blocks.itertuples()), batch_size=batch_size,
        collate_fn=functools.partial(do_collate, defaults=defaults, instr_params=instr_params, global_params=global_params, shuffle=shuffle),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    ema_loss = 0
    ema_idx = 0
    ema_beta = 0.995

    if opt_alg.lower() == 'adam':
        opt = torch.optim.Adam(list(instr_params.parameters()) + [global_params], lr=opt_alpha,  betas=(opt_beta, opt_beta_2))
    elif opt_alg.lower() == 'sgd':
        opt = torch.optim.SGD(list(instr_params.parameters()) + [global_params], lr=opt_alpha, momentum=opt_beta, nesterov=opt_nesterov)

    opt_lstm = torch.optim.Adam(lstm.parameters())

    n_epoch = 0
    if not epochs:
        epochs = float('inf')
    while n_epoch < epochs:
        if validate:
            break
        with tqdm.tqdm(total=len(blocks)) as pbar:
            for batch in train_dataloader:
                if (n_dataitems // batch_size) % 100 == 0:
                    for s in ['latest', time.time()]:
                        torch.save(
                            (n_dataitems, train_losses, instr_params.state_dict(), global_params.detach()),
                            get_name(s)
                        )
                preds = lstm(
                    batch.instrs.to(device, non_blocking=True),
                    batch.instr_lens,
                    batch.block_lens,
                    batch.instr_params.to(device, non_blocking=True),
                    batch.global_params.to(device, non_blocking=True),
                ).reshape(-1)
                ys = batch.timings.to(device, non_blocking=True)
                err = ((preds - ys).abs() / ys).mean()

                opt.zero_grad()
                opt_lstm.zero_grad()
                err.backward()
                opt.step()
                n_dataitems += batch_size

                ema_loss = ema_loss * ema_beta + err.item() * (1 - ema_beta)
                ema_idx += 1
                current_loss = ema_loss / (1 - ema_beta**ema_idx)

                pbar.update(batch_size)
                pbar.set_postfix(loss=current_loss)
                if len(train_losses) == 0 or n_dataitems >= train_losses[-1][0] + batch_size*50:
                    train_losses.append((n_dataitems, time.time() - start_time + elapsed_time, current_loss))
        n_epoch += 1
    if not validate:
        for s in ['latest', time.time()]:
            torch.save(
                (n_dataitems, train_losses, instr_params.state_dict(), global_params.detach()),
                get_name(s)
            )


    # test
    ema_loss = 0
    ema_idx = 0
    ema_beta = 0.995
    all_preds = []
    all_errs = []
    with tqdm.tqdm(total=len(blocks)) as pbar:
        with torch.no_grad():
            for batch in train_dataloader:
                preds = lstm(
                    batch.instrs.to(device, non_blocking=True),
                    batch.instr_lens,
                    batch.block_lens,
                    batch.instr_params.to(device, non_blocking=True),
                    batch.global_params.to(device, non_blocking=True),
                ).reshape(-1)
                ys = batch.timings.to(device, non_blocking=True)
                err = ((preds - ys).abs() / ys).mean()

                ema_loss = ema_loss * ema_beta + err.item() * (1 - ema_beta)
                all_preds.append(preds.detach().cpu().numpy())
                all_errs.append(err.item())
                ema_idx += 1
                current_loss = ema_loss / (1 - ema_beta**ema_idx)

                pbar.update(batch_size)
                pbar.set_postfix(loss=current_loss)

def extract_parameters(name, sim, arch, model_name, params_name):
    version = 'latest'

    sample_params = read(name, 'sample-params')
    blocks = read(name, 'blocks').reset_index()

    defaults = read(name, 'default_params')[arch]
    all_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if group_port_re.match(c)]
    m_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if (group_port_re if sample_params['use_groups'] else single_port_re).match(c)]
    readadvance_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if 'readadvance' in c]
    defaults = SampleTableResult(
        instruction_table=defaults.instruction_table[all_port_cols + readadvance_cols + ['NumMicroOps', 'Latency']],
        global_vector=defaults.global_vector,
    )

    all_idxs = list(blocks['idx'])
    np.random.RandomState(0).shuffle(all_idxs)
    train_idxs = all_idxs[:int(0.8 * len(all_idxs))]
    val_idxs = all_idxs[int(0.8 * len(all_idxs)):int(0.9 * len(all_idxs))]
    test_idxs = all_idxs[int(0.9 * len(all_idxs)):]
    blocks = blocks[blocks['idx'].isin(set(train_idxs))]

    base_dir = '{}/{}'.format(DATA_BASE, name)
    model_file_name = '{}-model'.format(model_name)
    def get_name(suffix):
        if params_name is None:
            return os.path.join(base_dir, '{}-params-{}'.format(model_file_name, suffix))
        else:
            return os.path.join(base_dir, '{}-params-{}-{}'.format(model_file_name, params_name, suffix))

    instr_params = torch.nn.Embedding(20000, len(all_port_cols) + 8)
    instr_params.weight = torch.nn.Parameter(torch.Tensor(defaults.instruction_table.values.astype(np.float32)), requires_grad=True)
    global_params = torch.nn.Parameter(torch.FloatTensor(defaults.global_vector), requires_grad=True)

    (n_dataitems, train_losses, instr_params_sd, global_params_sd) = torch.load(get_name('latest'))
    instr_params.load_state_dict(instr_params_sd)
    global_params.data.copy_(global_params_sd)

    defaults.instruction_table.loc[:, :] = instr_params.weight.round().abs().int().numpy()
    defaults.global_vector[:] = global_params.round().abs().int().numpy()
    shutil.move(write_params_to_file(defaults)[1], get_name('extracted'))

def validate_parameters(name, sim, arch, model_name, params_name, sample_size):
    version = 'latest'

    sample_params = read(name, 'sample-params')
    blocks = read(name, 'blocks').reset_index()

    defaults = read(name, 'default_params')[arch]
    all_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if group_port_re.match(c)]
    m_port_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if (group_port_re if sample_params['use_groups'] else single_port_re).match(c)]
    readadvance_cols = [c for (i, c) in enumerate(defaults.instruction_table.columns) if 'readadvance' in c]
    defaults = SampleTableResult(
        instruction_table=defaults.instruction_table[all_port_cols + readadvance_cols + ['NumMicroOps', 'Latency']],
        global_vector=defaults.global_vector,
    )

    all_idxs = list(blocks['idx'])
    np.random.RandomState(0).shuffle(all_idxs)
    train_idxs = all_idxs[:int(0.8 * len(all_idxs))]
    val_idxs = all_idxs[int(0.8 * len(all_idxs)):int(0.9 * len(all_idxs))]
    test_idxs = all_idxs[int(0.9 * len(all_idxs)):]
    blocks = blocks[blocks['idx'].isin(set(test_idxs))]

    base_dir = '{}/{}'.format(DATA_BASE, name)
    model_file_name = '{}-model'.format(model_name)
    def get_name(suffix):
        if params_name is None:
            return os.path.join(base_dir, '{}-params-{}'.format(model_file_name, suffix))
        else:
            return os.path.join(base_dir, '{}-params-{}-{}'.format(model_file_name, params_name, suffix))


    if sim == 'mca':
        handle = McaHandle(arch)
    elif sim == 'exegesis':
        handle = ExegesisHandle()

    if sample_size and sample_size < len(blocks):
        blocks = blocks.sample(sample_size)
    tru = blocks['{}-true'.format(long_to_short[arch])]
    with open(get_name('extracted')) as f:
        env = {l.strip().split()[0]: l.strip().split()[1] for l in f}


    pts = ['HWDivider', 'HWFPDivider', 'HWPort0', 'HWPort01', 'HWPort015', 'HWPort0156', 'HWPort04', 'HWPort05', 'HWPort056', 'HWPort06', 'HWPort1', 'HWPort15', 'HWPort16', 'HWPort2', 'HWPort23', 'HWPort237', 'HWPort3', 'HWPort4', 'HWPort5', 'HWPort56', 'HWPort6', 'HWPort7', 'HWPortAny']
    def minize(used_opcodes, env):
     res = {}
     def cp(key):
         if key in env:
             res[key] = env[key]
     cp('dispatch-width')
     cp('microop-buffer-size')
     for opcode in used_opcodes:
         cp('latency-{}-0'.format(opcode))
         cp('microops-{}'.format(opcode))
         for port in pts:
             cp('port-{}-{}'.format(opcode, port))
         for r in range(8):
             cp('readadvance-{}-{}-0'.format(opcode, r))
     return res


    def timeit(b):
        code, mcinsts = b
        return handle.get_timing_fast(
            minize(mcinsts.split(), env),
            code
        )

    with mp.Pool() as p:
        blocks['pred'] = np.array(list(tqdm.tqdm(p.imap(timeit, blocks[['code', 'mcinsts']].values, chunksize=64), total=len(blocks))))
    res = blocks['pred']
    print('Error: {}, Corr: {}'.format(
        ((res - tru) / tru).abs().mean(),
        res.corr(tru, method='kendall'),
    ))

def read(name, task):
    base_dir = '{}/{}'.format(DATA_BASE, name)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, task), 'rb') as f:
        return pickle.load(f)

def write(name, task, result):
    base_dir = '{}/{}'.format(DATA_BASE, name)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, task), 'wb') as f:
        pickle.dump(result, f)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Gather data for training')
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--task', required=True, help='Task')
    parser.add_argument('--sim', choices=['mca', 'exegesis'], required=False, help='Simulator used during sampling or training')
    parser.add_argument('--arch', choices=long_arches, required=False, help='Architecture used during sampling or training')
    parser.add_argument('--model-name', help='Modelname')
    parser.add_argument('--model-width', type=int, default=256)
    parser.add_argument('--model-depth', type=int, default=4)
    parser.add_argument('--model-bidirectional', default=False, type=str2bool)
    parser.add_argument('--approx-max-delta', type=float, default=np.inf)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--opt-alg', default='adam')
    parser.add_argument('--opt-alpha', type=float, default=1e-3)
    parser.add_argument('--opt-beta', type=float, default=0.9)
    parser.add_argument('--opt-beta-2', type=float, default=0.999)
    parser.add_argument('--opt-nesterov', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--validate')
    parser.add_argument('--params-name')
    parser.add_argument('--timing-arch')
    parser.add_argument('--epochs', type=int, default=None)


    parser.add_argument('--n-forks', required=False, type=int)
    parser.add_argument('--sample-shuffle', type=str2bool, default=False, required=False)
    parser.add_argument('--sample-params-min-latency', type=int, default=0, required=False)
    parser.add_argument('--sample-params-max-latency', type=int, default=5, required=False)
    parser.add_argument('--sample-params-min-n-uops', type=int, default=1, required=False)
    parser.add_argument('--sample-params-max-n-uops', type=int, default=10, required=False)
    parser.add_argument('--sample-params-min-n-ports', type=int, default=0, required=False)
    parser.add_argument('--sample-params-max-n-ports', type=int, default=2, required=False)
    parser.add_argument('--sample-params-min-port-time', type=int, default=0, required=False)
    parser.add_argument('--sample-params-max-port-time', type=int, default=2, required=False)
    parser.add_argument('--sample-params-min-dispatch-width', type=int, default=1, required=False)
    parser.add_argument('--sample-params-max-dispatch-width', type=int, default=10, required=False)
    parser.add_argument('--sample-params-min-microop-buffer-size', type=int, default=50, required=False)
    parser.add_argument('--sample-params-max-microop-buffer-size', type=int, default=250, required=False)
    parser.add_argument('--sample-params-min-readadvance', type=int, default=0, required=False)
    parser.add_argument('--sample-params-max-readadvance', type=int, default=5, required=False)
    parser.add_argument('--sample-params-min-microops', type=int, default=0, required=False)
    parser.add_argument('--sample-params-max-microops', type=int, default=5, required=False)
    parser.add_argument('--sample-params-by-schedclass', type=str2bool, default=False, required=False)
    parser.add_argument('--sample-params-use-groups', type=str2bool, default=False, required=False)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    if args.task == 'blocks':
        blocks = get_blocks()
        write(args.name, args.task, blocks)
    elif args.task == 'default_params':
        defaults = get_default_params()
        write(args.name, args.task, defaults)
    elif args.task == 'default_timings':
        blocks = read(args.name, 'blocks')
        defaults = read(args.name, 'default_params')
        default_timings = get_default_timings(blocks, defaults)
        write(args.name, args.task, default_timings)
    elif args.task == 'sample_timings':
        assert args.sim is not None
        if args.sim != 'exegesis':
            assert args.arch is not None
        sample_table_params = {k[len('sample_params_'):]: v for (k, v) in args._get_kwargs() if k.startswith('sample_params_')}
        write_sample_timings(args.name, args.sim, args.arch, sample_table_params, args.n_forks, args.sample_shuffle)
    elif args.task == 'approximation':
        train_approximation(args.name, args.sim, args.arch, args.model_name, args.model_width, args.model_depth, args.model_bidirectional, args.batch_size, args.approx_max_delta, args.sample_shuffle, args.opt_alg, args.opt_alpha, args.opt_beta, args.opt_beta_2, args.opt_nesterov, args.device, args.sample_size, args.validate, args.epochs)
    elif args.task == 'parameters':
        train_parameters(args.name, args.sim, args.arch, args.model_name, args.batch_size, args.sample_shuffle, args.seed, args.params_name, args.opt_alg, args.opt_alpha, args.opt_beta, args.opt_beta_2, args.opt_nesterov, args.device, args.timing_arch, args.validate, args.epochs)
    elif args.task == 'extract':
        extract_parameters(args.name, args.sim, args.arch, args.model_name, args.params_name)
    elif args.task == 'validate':
        validate_parameters(args.name, args.sim, args.arch, args.model_name, args.params_name, args.sample_size)
    else:
        raise ValueError('Unknown task "{}"'.format(args.task))

if __name__ == '__main__':
    main()
