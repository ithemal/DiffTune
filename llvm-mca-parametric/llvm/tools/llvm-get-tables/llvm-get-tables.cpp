//===-- llvm-mca.cpp - Machine Code Analyzer -------------------*- C++ -* -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility is a simple driver that allows static performance analysis on
// machine code similarly to how IACA (Intel Architecture Code Analyzer) works.
//
//   llvm-mca [options] <file-name>
//      -march <type>
//      -mcpu <cpu>
//      -o <file>
//
// The target defaults to the host target.
// The cpu defaults to the 'native' host cpu.
// The output defaults to standard output.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/Stages/EntryStage.h"
#include "llvm/MCA/Stages/InstructionTables.h"
#include "llvm/MCA/Support.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include "llvm/MCA/InstrBuilder.h"
#include "llvm/Support/Parameters.h"

#include <random>
#include <unordered_map>

using namespace llvm;

static cl::opt<std::string>
    ArchName("march", cl::desc("Target architecture. "
                               "See -version for available targets"));

static cl::opt<std::string>
    TripleName("mtriple",
               cl::desc("Target triple. See -version for available targets"));

static cl::opt<std::string>
    MCPU("mcpu",
         cl::desc("Target a specific cpu type (-mcpu=help for details)"),
         cl::value_desc("cpu-name"), cl::init("native"));

enum SampleType {
                 builtin, sample
};

cl::opt<SampleType> sampleType(cl::desc("Whether to use builtin or sampled params"),
  cl::values(
    clEnumVal(builtin , "Builtin"),
    clEnumVal(sample, "Sampled")));

static cl::opt<int>
    SampleSeed("seed",
               cl::desc("Sample seed. -1(default) for random"),
               cl::init(-1)
               );

static cl::opt<int> MinLatency("min-latency",
                               cl::desc("Min Sample Latency"),
                               cl::init(0)
                               );
static cl::opt<int> MaxLatency("max-latency",
                               cl::desc("Max Sample Latency"),
                               cl::init(10)
                               );

static cl::opt<int> MinUops("min-uops",
                               cl::desc("Min Sample Uops"),
                               cl::init(1)
                               );
static cl::opt<int> MaxUops("max-uops",
                               cl::desc("Max Sample Uops"),
                               cl::init(10)
                               );

static cl::opt<int> MinPorts("min-ports",
                               cl::desc("Min Sample Ports"),
                               cl::init(0)
                               );
static cl::opt<int> MaxPorts("max-ports",
                               cl::desc("Max Sample Ports"),
                               cl::init(5)
                               );

static cl::opt<int> MinPortTime("min-port-time",
                               cl::desc("Min Sample Port-Time"),
                               cl::init(0)
                               );
static cl::opt<int> MaxPortTime("max-port-time",
                               cl::desc("Max Sample Port-Time"),
                               cl::init(5)
                               );

static cl::opt<bool> BySchedClass("by-schedclass",
                               cl::desc("Sample by schedclass instead of intr"),
                               cl::init(false)
                               );



static SmallVector<uint64_t, 8> M_Masks;
static bool is_unrolled_initialized = false;
static uint64_t NormMask(const MCSchedModel *SchedModel, unsigned I) {
  const MCProcResourceDesc *Desc = SchedModel->getProcResource(I);
  if (Desc->SubUnitsIdxBegin) {
    for (int J = 63; J >= 0; J--) {
      if (M_Masks[I] & (1ULL << J)) {
        return M_Masks[I] ^ (1ULL << J);
      }
    }
  }
  return M_Masks[I];
}
static void readUnrolledCycles(const MCSchedModel *SchedModel) {
  if (!is_unrolled_initialized) {
    M_Masks.resize(SchedModel->getNumProcResourceKinds());
    unsigned ProcResourceID = 0;
    M_Masks[0] = 0;
    for (unsigned I = 1, E = SchedModel->getNumProcResourceKinds(); I < E; ++I) {
      const MCProcResourceDesc *Desc = SchedModel->getProcResource(I);
      if (Desc->SubUnitsIdxBegin)
        continue;
      M_Masks[I] = 1ULL << ProcResourceID;
      ProcResourceID++;
    }
    for (unsigned I = 1, E = SchedModel->getNumProcResourceKinds(); I < E; ++I) {
      const MCProcResourceDesc *Desc = SchedModel->getProcResource(I);
      if (!Desc->SubUnitsIdxBegin)
        continue;
      M_Masks[I] = 1ULL << ProcResourceID;
      for (unsigned U = 0; U < Desc->NumUnits; ++U) {
        uint64_t OtherMask = M_Masks[Desc->SubUnitsIdxBegin[U]];
        M_Masks[I] |= OtherMask;
      }
      ProcResourceID++;
    }

    is_unrolled_initialized = true;
  }
}



namespace {

const Target *getTarget(const char *ProgName) {
  if (TripleName.empty())
    TripleName = Triple::normalize(sys::getDefaultTargetTriple());
  Triple TheTriple(TripleName);

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, TheTriple, Error);
  if (!TheTarget) {
    errs() << ProgName << ": " << Error;
    return nullptr;
  }

  // Return the found target.
  return TheTarget;
}
} // end of anonymous namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // // Initialize targets and assembly parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();

  // // Enable printing of available targets when flag --version is specified.

  // Parse flags and initialize target options.
  cl::ParseCommandLineOptions(argc, argv,
                              "llvm machine code performance analyzer.\n");

  // Get the target from the triple. If a triple is not specified, then select
  // the default triple for the host. If the triple doesn't correspond to any
  // registered target, then exit with an error message.
  const char *ProgName = argv[0];
  const Target *TheTarget = getTarget(ProgName);
  if (!TheTarget)
    return 1;

  // GetTarget() may replaced TripleName with a default triple.
  // For safety, reconstruct the Triple object.
  Triple TheTriple(TripleName);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  assert(MRI && "Unable to create target register info!");

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
  assert(MAI && "Unable to create target asm info!");

  std::unique_ptr<buffer_ostream> BOS;

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());

  std::unique_ptr<MCInstrAnalysis> MCIA(
      TheTarget->createMCInstrAnalysis(MCII.get()));

  if (!MCPU.compare("native"))
    MCPU = llvm::sys::getHostCPUName();

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, /* FeaturesStr */ ""));
  if (!STI->isCPUStringValid(MCPU))
    return 1;

  if (!STI->getSchedModel().hasInstrSchedModel()) {
    WithColor::error()
        << "unable to find instruction-level scheduling information for"
        << " target triple '" << TheTriple.normalize() << "' and cpu '" << MCPU
        << "'.\n";

    if (STI->getSchedModel().InstrItineraries)
      WithColor::note()
          << "cpu '" << MCPU << "' provides itineraries. However, "
          << "instruction itineraries are currently unsupported.\n";
    return 1;
  }


  const MCSchedModel &SM = STI->getSchedModel();

  std::cout << "dispatch-width " << SM.IssueWidth << std::endl;
  std::cout << "microop-buffer-size " << SM.MicroOpBufferSize << std::endl;

  readUnrolledCycles(&SM);

  std::random_device r;
  auto rnd = r();
  if (SampleSeed != -1) {
    rnd = SampleSeed;
  }
  std::default_random_engine eng(rnd);
  std::uniform_int_distribution<> latency_dist(MinLatency, MaxLatency);
  std::uniform_int_distribution<> uops_dist(MinUops, MaxUops);
  std::uniform_int_distribution<> n_ports_dist(MinPorts, MaxPorts);
  std::uniform_int_distribution<> port_idx_dist(1, SM.getNumProcResourceKinds());
  std::uniform_int_distribution<> port_time_dist(MinPortTime, MaxPortTime);

  if (BySchedClass) {
    std::cerr << "By Schedclass not yet supported" << std::endl;
    return 1;
  }


  for (unsigned Opcode = 1; Opcode < MCII->getNumOpcodes(); Opcode++) {
    const MCInstrDesc &MCDesc = MCII->get(Opcode);
    unsigned SchedClassID = MCDesc.getSchedClass();

    std::cout << "opcode-name " << MCII->getName(Opcode).data() << std::endl;

    const MCSchedClassDesc SC = *SM.getSchedClassDesc(SchedClassID);
    if (sampleType == sample) {
      std::cout << "latency-" << Opcode << "-0 " << latency_dist(eng) << std::endl;
      std::cout << "microops-" << Opcode << " " << uops_dist(eng) << std::endl;

      auto n_ports = n_ports_dist(eng);

      static std::unordered_map<int, int> portTimeMap;
      for (int i = 0; i < n_ports; i++) {
        portTimeMap[port_idx_dist(eng)] = port_time_dist(eng);
      }
      for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
        const MCProcResourceDesc *Desc = SM.getProcResource(I);
        auto res = portTimeMap.find(I);
        auto tim = 0;
        if (res != portTimeMap.end()) {
          tim = res->second;
        }
        std::cout << "port-" << Opcode << "-" << Desc->Name << " " << tim << std::endl;
      }

      continue;
    }



    if (SC.NumWriteLatencyEntries) {
      const MCWriteLatencyEntry *WLEntry = STI->getWriteLatencyEntry(&SC, 0);
      std::cout << "latency-" << Opcode << "-0 " << WLEntry->Cycles << std::endl;
    } else {
      std::cout << "latency-" << Opcode << "-0 1" << std::endl;
    }

    if (SC.NumMicroOps > 1000) {
      std::cout << "microops-" << Opcode << " " << 1 << std::endl;
    } else {
      std::cout << "microops-" << Opcode << " " << SC.NumMicroOps << std::endl;
    }



    static SmallVector<int, 8> cycles;
    cycles.resize(SM.getNumProcResourceKinds());
    for (unsigned I = 0; I < SM.getNumProcResourceKinds(); I++) {
      cycles[I] = 0;
    }

    auto PI = STI->getWriteProcResBegin(&SC);
    auto PE = STI->getWriteProcResEnd(&SC);
    for (; PI != PE; ++PI) {
      auto I = PI->ProcResourceIdx;
      cycles[I] = PI->Cycles;
    }

    for (unsigned I = 1; I < SM.getNumProcResourceKinds(); I++) {
      for (unsigned J = I + 1; J < SM.getNumProcResourceKinds(); J++) {
        if ((NormMask(&SM, I) & NormMask(&SM, J)) == NormMask(&SM, I)) {
          cycles[J] -= cycles[I];
        }
      }
    }

    for (unsigned I = 1; I < SM.getNumProcResourceKinds(); I++) {
      const MCProcResourceDesc *Desc = SM.getProcResource(I);
      std::cout << "port-" << Opcode << "-" << Desc->Name << " " << cycles[I] << std::endl;
    }

    for (unsigned int I = 0; I < 7; I++) {
      std::cout << "readadvance-" << Opcode << "-" << I << "-0 " << STI->getReadAdvanceCycles(&SC, I, 0) << "\n";
    }
  }

  return 0;
}
