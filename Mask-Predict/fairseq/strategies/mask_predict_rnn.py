# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens
from nat_merge import GreedyMerger

@register_strategy('mask_predict_rnn')
class MaskPredictRNN(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.total_context = args.len_context + 1
        self.args = args

    def tile_length(self, target, fill):

        max_length = max([len(row) for row in target])
        target = [t + [fill] * (max_length - len(t)) for t in target]
        return torch.tensor(target)

    def merger(self, tgt_tokens, token_probs, seq_lens, merger, pad_idx, mask_idx, as_final_output=False, keep_rate=0.5):
        bsz, seq_len = tgt_tokens.size()
        merger_token_probs = token_probs.log().view(bsz, -1, self.total_context).cpu().numpy()
        merger_tgt_tokens = tgt_tokens.view(bsz, -1, self.total_context).cpu().numpy()
        merger_tgt_rst = []
        seq_len = merger_tgt_tokens.shape[1]
        for tt, tp, tl in zip(merger_tgt_tokens, merger_token_probs, seq_lens):
            ret_tokens, orig_tokens = merger.run(tt[:tl], scores=tp[:tl],
                                                 as_final_output=as_final_output, keep_rate=keep_rate)
            # if not as_final_output:
            #     merger_tgt_rst.append(torch.cat((tgt_tokens.new_tensor(ret_tokens),
            #         tgt_tokens.new_ones(seq_len - len(ret_tokens)) * pad_idx)))
            # else:
            #     merger_tgt_rst.append(ret_tokens) 
            merger_tgt_rst.append(ret_tokens)
        max_length = max([len(row) for row in merger_tgt_rst])
        target = [t + [pad_idx] * (max_length - len(t)) for t in merger_tgt_rst]
        return tgt_tokens.new(target)
        # if not as_final_output:
        #     merger_tgt_rst = torch.stack(merger_tgt_rst, dim=0)
        #     return merger_tgt_rst
        # else:
        #     return self.tile_length(merger_tgt_rst, pad_idx)

    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        # tgt_tokens = golden_tgt[:,0::3]
        self.args.win_size = self.total_context
        self.args.merge_k = self.total_context
        merger = GreedyMerger(self.args, tgt_dict.bos(), tgt_dict.eos(), tgt_dict.mask())
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        tgt_tokens_list = []
        token_probs_list = []
        old_tgt_tokens = tgt_tokens
        iterations = seq_len if self.iterations is None else self.iterations
        with open('intermedidate.txt', 'a') as f:
            for target_token in tgt_tokens:
                target_str = tgt_dict.string(target_token, '@@ ', escape_unk=True, remove_eos=False)
                f.write('input:' + target_str + '\n')
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        reshape_tgt_tokens = tgt_tokens.view(-1, self.total_context)
        # with open('intermedidate.txt', 'a') as f:
        #     for target_token in reshape_tgt_tokens:
        #         target_str = tgt_dict.string(target_token, '', escape_unk=True, remove_eos=False)
        #         f.write('window:' + target_str + '\n')
        tgt_tokens = self.merger(tgt_tokens, token_probs,
                                seq_lens, merger, tgt_dict.pad(), tgt_dict.mask(),
                                keep_rate=1./iterations, as_final_output=iterations==1)
        # assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        # with open('intermedidate.txt', 'a') as f:
        #     for target_token in tgt_tokens:
        #         target_str = tgt_dict.string(target_token, '@@ ', escape_unk=True, remove_eos=False)
        #         f.write('output: ' + target_str + '\n')
        for counter in range(1, iterations):
            # tgt_tokens_list.append([convert_tokens(tgt_dict, tgt_tokens[i]) for i in range(tgt_tokens.size()[0])])
            decoder_out = model.decoder.generate(tgt_tokens, encoder_out)
            tgt_tokens, token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            reshape_tgt_tokens = tgt_tokens.view(-1, self.total_context)
            # with open('intermedidate.txt', 'a') as f:
            #     for target_token in reshape_tgt_tokens:
            #         target_str = tgt_dict.string(target_token, '', escape_unk=True, remove_eos=False)
            #         f.write('window: ' + target_str + '\n')
            tgt_tokens = self.merger(tgt_tokens, token_probs,
                                    seq_lens, merger, tgt_dict.pad(), tgt_dict.mask(),
                                    as_final_output=counter+1==iterations,
                                    keep_rate=1./iterations * (counter+1))
            # with open('intermedidate.txt', 'a') as f:
            #     for target_token in tgt_tokens:
            #         target_str = tgt_dict.string(target_token, '@@ ', escape_unk=True, remove_eos=False)
            #         f.write('output: ' + target_str + '\n')
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder.generate(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

