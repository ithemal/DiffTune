import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ParamLstm(nn.Module):
    def __init__(self, n_tokens, hidden_size, instr_param_size, global_param_size, n_layers, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, hidden_size)
        self.char_bidirectional = bidirectional
        self.word_bidirectional = bidirectional
        self.char_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=self.char_bidirectional, batch_first=True)
        self.word_lstm = nn.LSTM(hidden_size * (1 + self.char_bidirectional) + instr_param_size + global_param_size, hidden_size, n_layers, bidirectional=self.word_bidirectional, batch_first=True)
        self.final_layer = nn.Linear(hidden_size * (1 + self.word_bidirectional) + global_param_size, 1)

    def forward(self, tokens, token_lens, block_lens, instr_params, global_params):
        batch_size = len(block_lens)

        chars_packed = nn.utils.rnn.pack_padded_sequence(self.embedding(tokens), token_lens, batch_first=True, enforce_sorted=False)
        _, (instrs, _) = self.char_lstm(chars_packed)
        instrs = torch.cat(list(map(lambda x: x.squeeze(0), torch.split(instrs[-(1 + self.char_bidirectional):, :], 1, dim=0))), dim=-1)
        instrs = torch.cat((instrs, instr_params), dim=1)

        res = []
        idx = 0
        for block_idx, block_len in enumerate(block_lens):
            blockres = []
            for i in range(block_len):
                blockres.append(instrs[idx])
                idx += 1

            blockres = torch.stack(blockres)
            m_global_params = global_params[block_idx].reshape(1, -1).expand(block_len, -1)
            blockres = torch.cat((blockres, m_global_params), dim=-1)
            res.append(blockres)

        instrs = nn.utils.rnn.pack_sequence(res, enforce_sorted=False)
        _, (blocks, _) = self.word_lstm(instrs)
        blocks = torch.cat(list(map(lambda x: x.squeeze(0), torch.split(blocks[-(1 + self.word_bidirectional):, :], 1, dim=0))), dim=-1)

        final_value = torch.cat((blocks, global_params.view(batch_size, -1)), dim=-1)
        return self.final_layer(final_value)
