# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import random

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i] for s in samples], pad_idx, eos_idx, left_pad=False,
                ))
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
            )

    is_target_list = isinstance(samples[0]['dec_ctx_src'], list)
    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(s['ntokens'] for s in samples),
        'net_input': {
            'src_tokens': merge('enc_source'),
            'src_lengths': torch.LongTensor([
                s['enc_source'].numel() for s in samples
            ]),
            'prev_output_tokens': merge('dec_source'),
            'ctx_src': merge('dec_ctx_src', is_target_list)
        },
        'target': merge('dec_ctx_tgt', is_target_list),
        'ori_output_tokens': merge('ori_dec_source'),
        'target_ratio': merge('txt_loss_ratio', is_target_list),
        'nsentences': samples[0]['enc_source'].size(0),
    }

    """id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    
    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch"""


class LanguagePairRNNContextMask(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=2048, max_target_positions=2048,
        shuffle=True, input_feeding=True,
        dynamic_length=False,
        mask_range=False,
        train=True,
        seed=None,
        len_context=None,
        context_dir='both',
        unmask_loss_ratio=0.1,
        del_ratio=0.15
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.dynamic_length = dynamic_length
        self.mask_range = mask_range
        self.train = train
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.seed = seed
        self.len_context = len_context
        self.context_dir = context_dir
        self.unmask_loss_ratio = unmask_loss_ratio
        self.del_ratio = del_ratio

    def __getitem__(self, index):
        enc_source, dec_source, dec_ctx_src, dec_ctx_tgt, txt_loss_ratio, ntokens, ori_dec_source = self._make_source_target(self.src[index], self.tgt[index])
        return {'id': index, 'enc_source': enc_source, 'dec_source': dec_source,
                'dec_ctx_src': dec_ctx_src, 'dec_ctx_tgt': dec_ctx_tgt,
                'ntokens': ntokens, 'txt_loss_ratio': txt_loss_ratio, 'ori_dec_source': ori_dec_source}

    def __len__(self):
        return len(self.src)


    def _insert_delete_mask(self, source, del_list, ins_list, scale=1, fill=None):
        for ins_idx in reversed(ins_list):
            if fill is None:
                source = torch.cat((source[:(ins_idx + 1) * scale],
                                    source[ins_idx * scale:(ins_idx + 1) * scale],
                                    source[(ins_idx + 1) * scale:]))
            else:
                insert_item = source.new([fill] * scale)
                source = torch.cat((source[:(ins_idx + 1) * scale],
                                    insert_item,
                                    source[(ins_idx + 1) * scale:]))
        for del_idx in reversed(del_list):
            source = torch.cat((source[:(del_idx) * scale], source[(del_idx+1) * scale:]))
        # acc = 1
        # new_idx = []
        # for i in range(len(idx)):
        #     new_idx.append(idx[i] + acc)
        #     # source.insert(idx[i] + acc, source[idx[i] + acc - 1])
        #     source = torch.cat((source[:idx[i] + acc],
        #         source[idx[i] + acc - 1].reshape(1), source[idx[i] + acc:]))
        #     acc += 1
        return source

    # def _delete_mask(self, source, idxs, scale=1):
    #     for idx in reversed(idxs):
    #         source = torch.cat((source[:idx], source[(idx+1) * scale:]))
    #     return source


    def _update_index(self, delete_idx, insert_idx):
        temp_delete_idx = np.copy(delete_idx)
        for idx in insert_idx:
            delete_idx[temp_delete_idx > idx] += 1
        return delete_idx

    def _make_source_target(self, source, target):
        if self.dynamic_length:
            max_len = 3 * len(source) // 2 + 1
            target = target.new((target.tolist() + ([self.tgt_dict.eos()] * (max_len - len(target))))[:max_len])
        
        min_num_masks = 1
        
        enc_source = source
        target = target.new([self.tgt_dict.bos()] + target.tolist())

        dec_source = target.new(target.tolist())
        dec_target_cp = target.new(target.tolist())
        dec_target = target.new([self.tgt_dict.pad()] * len(dec_source))
        len_ori_target = dec_target.size(0)
        ori_dec_source = dec_source
        if self.train:
            dec_target = dec_target_cp
            if min_num_masks < len(dec_source):
                sample_size = self.random.randint(min_num_masks, len(dec_source))
            else:
                sample_size = len(dec_source)
            if self.mask_range:
                start = self.random.randint(len(dec_source) - sample_size + 1)
                ind = list(range(start, start + sample_size))
            else:
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            max_del_size = round((len(dec_source) - 2) * self.del_ratio)
            del_mask = dec_source.new_ones(dec_source.size()).bool()
            if max_del_size > 0:
                del_size = self.random.randint(0, max_del_size)
                del_idx = self.random.choice(len(dec_source)-2 , size=del_size, replace=False) + 1 
                del_idx = dec_source.new(del_idx)
                del_mask[del_idx] = False
            dec_source[ind] = self.tgt_dict.mask()
            if self.context_dir == 'right':
                right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
                dec_target = torch.cat((dec_target, right_context))
                context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
                context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
            ctx_ind = ind * (self.len_context + 1)
            temp_ind = ctx_ind
            for i in range(self.len_context):
                ctx_ind = np.concatenate((ctx_ind, temp_ind - (i + 1) * self.len_context))
            ctx_ind = ctx_ind[ctx_ind >= 0]
            txt_loss_ratio = context_tgt.float().new([self.unmask_loss_ratio] * len(context_tgt))
            txt_loss_ratio[ctx_ind] = 1. - self.unmask_loss_ratio
            dec_source = torch.masked_select(dec_source, del_mask)
            scale_del_mask = del_mask.repeat(self.len_context+1).unsqueeze(1)
            
            scale_del_mask = scale_del_mask.view(self.len_context+1, -1).transpose(0, 1).contiguous().view(-1)
            context_src = torch.masked_select(context_src, scale_del_mask)
            context_tgt = torch.masked_select(context_tgt, scale_del_mask)
            txt_loss_ratio = torch.masked_select(txt_loss_ratio, scale_del_mask)
        else:
            dec_target = dec_target_cp
            dec_source[:] = self.tgt_dict.mask()
            if self.context_dir == 'both':
                left_context = dec_target.new([self.tgt_dict.bos()] * self.len_context)
                right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
                dec_target = torch.cat((left_context, dec_target, right_context))
                context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context*2])) for i in range(len_ori_target)])
                context_tgt = torch.cat([dec_target[i:i+self.len_context*2+1] for i in range(len_ori_target)])
            if self.context_dir == 'right':
                right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
                dec_target = torch.cat((dec_target, right_context))
                context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
                context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
            txt_loss_ratio = context_tgt.float().new([1.] * len(context_tgt))
        ntokens = dec_source.eq(self.tgt_dict.mask()).sum(-1).item()
        return enc_source, dec_source, context_src, context_tgt, txt_loss_ratio, ntokens, ori_dec_source

    # def _make_source_target(self, source, target):
    #     if self.dynamic_length:
    #         max_len = 3 * len(source) // 2 + 1
    #         target = target.new((target.tolist() + ([self.tgt_dict.eos()] * (max_len - len(target))))[:max_len])
        
    #     min_num_masks = 1
        
    #     enc_source = source
    #     target = target.new([self.tgt_dict.bos()] + target.tolist())

    #     dec_source = target.new(target.tolist())
    #     dec_target_cp = target.new(target.tolist())
    #     dec_target = target.new([self.tgt_dict.pad()] * len(dec_source))
    #     len_ori_target = dec_target.size(0)

    #     if self.train:
    #         dec_target = dec_target_cp
    #         if min_num_masks < len(dec_source):
    #             sample_size = self.random.randint(min_num_masks, len(dec_source))
    #         else:
    #             sample_size = len(dec_source)
    #         if self.mask_range:
    #             start = self.random.randint(len(dec_source) - sample_size + 1)
    #             ind = list(range(start, start + sample_size))
    #         else:
    #             ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
    #         delete_insert_probs = self.random.uniform(size=len_ori_target - 2)
            
    #         delete_idx = np.where(delete_insert_probs < 0.1)[0] + 1
    #         insert_idx = np.where(delete_insert_probs > 0.9)[0] + 1
    #         # print(delete_idx, insert_idx)
    #         delete_tokens = dec_source[delete_idx]
    #         insert_tokens = dec_source[insert_idx]
    #         # print("ori tokens", self.tgt_dict.string(dec_source))
    #         # print("delete tokens", self.tgt_dict.string(delete_tokens))
    #         # print("insert tokens", self.tgt_dict.string(insert_tokens))
            
    #         # ind = self._update_index(ind, insert_idx)
    #         delete_idx = self._update_index(delete_idx, insert_idx)
    #         # dec_target, insert_idx = self._insert_mask(dec_target, insert_idx)
    #         # ind = np.concatenate((ind, insert_idx))
    #         dec_source[ind] = self.tgt_dict.mask()
    #         if self.context_dir == 'right':
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
    #         ctx_ind = ind * (self.len_context + 1)
    #         temp_ind = ctx_ind
    #         for i in range(self.len_context):
    #             ctx_ind = np.concatenate((ctx_ind, temp_ind - (i + 1) * self.len_context))
    #         ctx_ind = ctx_ind[ctx_ind >= 0]
    #         txt_loss_ratio = context_tgt.float().new([self.unmask_loss_ratio] * len(context_tgt))
    #         txt_loss_ratio[ctx_ind] = 1. - self.unmask_loss_ratio
    #         dec_source = self._insert_delete_mask(dec_source, delete_idx, insert_idx, fill=self.tgt_dict.mask())
    #         context_src = self._insert_delete_mask(context_src, delete_idx, insert_idx, self.len_context+1)
    #         context_tgt = self._insert_delete_mask(context_tgt, delete_idx, insert_idx, self.len_context+1)
    #         txt_loss_ratio = self._insert_delete_mask(txt_loss_ratio, delete_idx, insert_idx, self.len_context+1)
    #         # print('update ori token: ', self.tgt_dict.string(dec_source))
    #         # print("context dec target tokens", self.tgt_dict.string(context_tgt, remove_eos=False))
    #         # exit()
    #     else:
    #         dec_target = dec_target_cp
    #         dec_source[:] = self.tgt_dict.mask()
    #         if self.context_dir == 'both':
    #             left_context = dec_target.new([self.tgt_dict.bos()] * self.len_context)
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((left_context, dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context*2])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context*2+1] for i in range(len_ori_target)])
    #         if self.context_dir == 'right':
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
    #         txt_loss_ratio = context_tgt.float().new([1.] * len(context_tgt))
    #     ntokens = dec_source.eq(self.tgt_dict.mask()).sum(-1).item()
    #     return enc_source, dec_source, context_src, context_tgt, txt_loss_ratio, ntokens

    # def _make_source_target(self, source, target):
    #     if self.dynamic_length:
    #         max_len = 3 * len(source) // 2 + 1
    #         target = target.new((target.tolist() + ([self.tgt_dict.eos()] * (max_len - len(target))))[:max_len])
        
    #     min_num_masks = 1
        
    #     enc_source = source
    #     target = target.new([self.tgt_dict.bos()] + target.tolist())
    #     dec_source = target.new(target.tolist())
    #     dec_target_cp = target.new(target.tolist())
    #     dec_target = target.new([self.tgt_dict.pad()] * len(dec_source))
        
    #     ntokens = dec_target.ne(self.tgt_dict.pad()).sum(-1).item()
    #     len_ori_target = dec_target.size(0)
    #     if self.train:
    #         if min_num_masks < len(dec_source):
    #             sample_size = self.random.randint(min_num_masks, len(dec_source))
    #         else:
    #             sample_size = len(dec_source)

    #         if self.mask_range:
    #             start = self.random.randint(len(dec_source) - sample_size + 1)
    #             ind = list(range(start, start + sample_size))
    #         else:
    #             ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
    #         dec_source[ind] = self.tgt_dict.mask()
    #         dec_target = dec_target_cp
    #         if self.context_dir == 'right':
    #             # left_context = dec_target.new([self.tgt_dict.bos()] * (self.len_context))
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
    #         ctx_ind = ind * (self.len_context + 1)
    #         temp_ind = ctx_ind
    #         for i in range(self.len_context):
    #             ctx_ind = np.concatenate((ctx_ind, temp_ind - (i + 1) * self.len_context))
    #         # print(ind, ctx_ind)
    #         ctx_ind = ctx_ind[ctx_ind >= 0]
    #         txt_loss_ratio = context_tgt.float().new([self.unmask_loss_ratio] * len(context_tgt))
    #         txt_loss_ratio[ctx_ind] = 1. - self.unmask_loss_ratio
            
    #     else:
    #         dec_target = dec_target_cp
    #         dec_source[:] = self.tgt_dict.mask()
    #         if self.context_dir == 'both':
    #             left_context = dec_target.new([self.tgt_dict.bos()] * self.len_context)
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((left_context, dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context*2])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context*2+1] for i in range(len_ori_target)])
    #         if self.context_dir == 'right':
    #             right_context = dec_target.new([self.tgt_dict.eos()] * (self.len_context))
    #             dec_target = torch.cat((dec_target, right_context))
    #             context_src = torch.cat([torch.cat((torch.tensor([self.tgt_dict.boseg()]), dec_target[i:i+self.len_context])) for i in range(len_ori_target)])
    #             context_tgt = torch.cat([dec_target[i:i+self.len_context+1] for i in range(len_ori_target)])
    #         txt_loss_ratio = context_tgt.float().new([1.] * len(context_tgt))
    #     # right_context_src = torch.cat([self.tgt_dict.boseg()] + [dec_target[i:i+args.len_context] for i in range(self.len_context, self.len_context + len_ori_target)])
    #     # right_context_tgt = torch.cat([dec_target[i:i+args.len_context+1] for i in range(self.len_context, self.len_context + len_ori_target)])
    #     # left_context_src = torch.cat([self.tgt_dict.boseg()] + [dec_target[i:i+args.len_context:-1] for i in range(0, len_ori_target)])
    #     # left_context_tgt = torch.cat([dec_target[i:i+args.len_context+1:-1] for i in range(0, len_ori_target)])
    #     # context_dec_target = torch.cat([dec_target[i:i + len_ori_target] for i in range(self.len_context * 2 + 1)], dim=0)
    #     # context_dec_target = context_dec_target.view(self.len_context * 2 + 1, -1).transpose(0, 1).contiguous().view(-1)
    #     # print ("masked tokens", self.tgt_dict.string(dec_source))
    #     # print ("original tokens", self.tgt_dict.string(dec_target, remove_eos=False), len(dec_target))
    #     # print("loss ratio", txt_loss_ratio)
    #     # print(ctx_ind)
    #     # print("context dec target tokens", self.tgt_dict.string(context_tgt, remove_eos=False))
    #     # exit()
    #     # print ("source tokens", self.src_dict.string(enc_source))
    #     return enc_source, dec_source, context_src, context_tgt, txt_loss_ratio, ntokens
               

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = num_tokens // max(src_len, tgt_len)

        enc_source, dec_source, dec_target, ntokens = self._make_source_target(self.src_dict.dummy_sentence(src_len), self.tgt_dict.dummy_sentence(tgt_len))

        return self.collater([
            {
                'id': i,
                'enc_source': enc_source,
                'dec_source': dec_source,
                'dec_target': dec_target,
                'ntokens': ntokens,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle and self.train and self.seed is None:
            return np.random.permutation(len(self))
        
        indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )
