#

from typing import List, Union, Set
import sys
import json
import numpy as np
from collections import Counter
import argparse
import heapq

np.random.seed(12345)

# =====
# algorithm helper

# performing aligning by matching scores
# -- input is 2d matrix of matching scores [len_a1, len_a2], should be >=0, 0 means absolute no match
# -- the order is important for breaking ties for add_a1,match,add_a2: by default prefer add-a2 later
def align_matches(match_score_arr, order_scores=None):
    DEFAULT_CODE = (0,1,2)  # a1/match/a2
    if order_scores is None:
        order_scores = DEFAULT_CODE
    assert np.all(match_score_arr>=0.)
    len1, len2 = match_score_arr.shape
    # recordings
    record_best_scores = np.zeros((1+len1, 1+len2), dtype=np.float32)  # best matching scores
    # pointers for back-tracing & also prefer order: by default add a1(x+=1); match; add a2(y+=1)
    record_best_codes = np.zeros((1+len1, 1+len2), dtype=np.int32)
    record_best_codes[0,:] = 2  # add a2 at y
    record_best_codes[:,0] = 0  # add a1 at x
    record_best_codes[0,0] = -1  # never used
    # loop: the looping order (ij or ji) does not matter
    for i in range(len1):
        ip1 = i + 1
        for j in range(len2):
            jp1 = j + 1
            s_match = match_score_arr[i,j] + record_best_scores[i,j]  # match one
            s_a1 = record_best_scores[i,jp1]  # add a1 on x
            s_a2 = record_best_scores[ip1,j]  # add a2 on y
            ordered_selections = sorted(zip((s_a1, s_match, s_a2), order_scores, DEFAULT_CODE))
            sel_score, _, sel_code = ordered_selections[-1]  # max score
            record_best_scores[ip1,jp1] = sel_score
            record_best_codes[ip1,jp1] = sel_code
    # backtracking for whole seq and aligning point
    # results of idx matches
    merge_to_a1, merge_to_a2 = [], []  # merge_idx -> a?_idx or None
    a1_to_merge, a2_to_merge = [], []  # a?_idx -> merge_idx
    back_i, back_j = len1, len2
    cur_midx = -1
    while back_i+back_j>0:  # there are still remainings
        code = record_best_codes[back_i, back_j]
        if code == 0:  # add a1[back_i-1]
            back_i -= 1
            merge_to_a1.append(back_i)
            merge_to_a2.append(None)
            a1_to_merge.append(cur_midx)
        elif code == 1:  # add matched a1[back_i-1],a2[back_j-1]
            back_i -= 1
            back_j -= 1
            merge_to_a1.append(back_i)
            merge_to_a2.append(back_j)
            a1_to_merge.append(cur_midx)
            a2_to_merge.append(cur_midx)
        elif code == 2:  # add a2[back_j-1]
            back_j -= 1
            merge_to_a1.append(None)
            merge_to_a2.append(back_j)
            a2_to_merge.append(cur_midx)
        else:
            raise NotImplementedError()
        cur_midx -= 1
    # reverse things
    merge_to_a1.reverse()
    merge_to_a2.reverse()
    merge_len = len(merge_to_a1)
    a1_to_merge = [merge_len+z for z in reversed(a1_to_merge)]
    a2_to_merge = [merge_len+z for z in reversed(a2_to_merge)]
    return merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge

# =====
# usually we want to match s2 to s1, thus hint is the start of s2 to s1
def align_seqs(s1, s2, hint_s2_on_s1_offset: float=None, hint_scale=0.1, match_f=(lambda x,y: float(x==y))):
    # first compare each pair to get match scores
    len1, len2 = len(s1), len(s2)
    match_score_arr = np.asarray([match_f(x,y) for x in s1 for y in s2]).reshape((len1, len2))
    if hint_s2_on_s1_offset is not None:
        assert hint_s2_on_s1_offset>=0 and hint_s2_on_s1_offset<len(s1), "Outside range of s1"
        posi_diff = np.arange(len1)[:, np.newaxis] - (np.arange(len2)+hint_s2_on_s1_offset)[np.newaxis, :]
        hint_rewards = hint_scale * np.exp(-np.abs(posi_diff))
        match_score_arr += (match_score_arr>0.).astype(np.float) * hint_rewards  # only add if >0
    # then get results
    return align_matches(match_score_arr)

# =====
# the merger

# one merged token
class MToken:
    def __init__(self, item, id, score):
        self.item = item  # the item itself
        self.ids = [id]  # list of ids
        self.scores = [score]  # list of scores
        # flags: no flags indicate inner portion in the segments
        self.flag_conflict = False  # conflicts between segs
        self.flag_gap = 0  # gap between segs (-1 means left, 1 means right)
        self.flag_link = False  # linking points between segs
        self.flag_delete = False  # flag for temp keeping but need to be deleted later
        # special one
        # self.flag_masked = False
        # score
        self._cached_score = None

    @property
    def score(self):
        if self._cached_score is None:
            self._cached_score = max(self.scores)  # simply maximum from scores, todo(+N): other choices?
        return self._cached_score

    def get_score_with_penalties(self, penalty_conflict, penalty_gap, penalty_link):
        return self.score \
               + penalty_conflict * int(self.flag_conflict) + penalty_gap * int(self.flag_gap!=0) \
               + penalty_link * int(self.flag_link)

    # direct adding
    def add(self, id, score):  # add matched one
        self.ids.append(id)
        self.scores.append(score)
        self._cached_score = None  # invalid cache

    # linking point
    def link(self, t1: 'MToken', flag_link=True):
        assert self.item == t1.item
        self.ids.extend(t1.ids)
        self.scores.extend(t1.scores)
        if flag_link:
            self.flag_link = True  # mark linking point
        self._cached_score = None  # invalid cache

    def flag_string(self):
        # return ''.join([(s if f else '') for f,s in zip(
        #     [self.flag_conflict, self.flag_gap, self.flag_link, self.flag_masked], "CGLM")])
        gap_s = "-" if self.flag_gap<0 else "+"
        return ''.join([(s if f else '') for f, s in zip([self.flag_conflict, self.flag_gap, self.flag_link], f"C{gap_s}L")])

    def to_string(self, verbose=False):
        flag_str = self.flag_string()
        ret = str(self.item)
        if flag_str:
            ret += f"[{flag_str}]"
        return ret

    def __repr__(self):
        return self.to_string()

# one segment
class MSeg:
    def __init__(self, tokens: List[MToken]):  # make a new seg
        # info
        self.tokens = tokens  # id: (window_idx, inner_idx)
        # # links
        # self.prev: MSeg = None
        # self.next: MSeg = None

    def __repr__(self):
        return str(self.tokens)

    # try to concat one seq, return success or not
    def concat(self, new_seq: List[MToken]) -> bool:
        new_len = len(new_seq)
        align_start = max(-new_len, -len(self.tokens))  # from which to start to try to align
        assert align_start<0, "Sth wrong since align_start is not negative!!"
        self_items = [z.item for z in self.tokens[align_start:]]
        new_items = [z.item for z in new_seq[:-align_start]]
        while align_start<0:  # while there are still room to concat
            if self_items[align_start:] == new_items[:-align_start]:  # ok to concat!
                for ii, ss in enumerate(new_seq):
                    align_ii = align_start + ii
                    if align_ii < 0:  # add for repeat ones
                        self.tokens[align_ii].link(ss, flag_link=False)  # no marking here!
                    else:  # add for new ones
                        self.tokens.append(ss)
                return True
            align_start += 1
        return False  # failed

    # merge another seg (inplaced): self[-K:] + seg1[:K]
    def merge(self, seg1: 'MSeg', merge_k: int, empty_score: float, resolve_conflict: bool, keep_conflict_merge: bool):
        to_merge_s0 = self.tokens[-merge_k:]
        to_merge_s1 = seg1.tokens[:merge_k]
        # first align these two
        align_res = align_seqs([z.item for z in to_merge_s0], [z.item for z in to_merge_s1])
        # then scan the results
        matched_pairs = [(a,b) for a,b in zip(align_res[0], align_res[1]) if (a is not None and b is not None)]
        if len(matched_pairs) == 0:  # no match, gap
            # mark gap and then simply concat
            to_merge_s0[-1].flag_gap = -1  # left get -1
            to_merge_s1[0].flag_gap = 1  # right get 1
            self.tokens.extend(seg1.tokens)
        else:  # linking with conflict solving
            merged_toks = []
            lpa, lpb = -1, -1
            for pa, pb in matched_pairs + [(None, None)]:
                piece0, piece1 = to_merge_s0[lpa+1:pa], to_merge_s1[lpb+1:pb]  # List[MToken], List[MToken]
                # special edge cases that are not marked as conflicts
                if lpa<0 and len(piece1)==0:
                    add_piece = piece0  # at the start of seg0
                elif pa is None and len(piece0)==0:
                    add_piece = piece1  # at the end of seg1
                elif resolve_conflict:
                    # add the conflicts, todo(+N): which side to choose and how to deal with empty?
                    score0 = np.mean([z.score for z in piece0]) if len(piece0)>0 else empty_score
                    score1 = np.mean([z.score for z in piece1]) if len(piece1)>0 else empty_score
                    if keep_conflict_merge:
                        add_piece = piece0 + piece1  # still add them all in order, but use different flags
                        if score0>=score1:
                            for z in piece0:
                                z.flag_conflict = True
                            for z in piece1:
                                z.flag_delete = True
                        else:
                            for z in piece1:
                                z.flag_conflict = True
                            for z in piece0:
                                z.flag_delete = True
                    else:  # keep only one piece
                        add_piece = piece0 if (score0>=score1) else piece1
                        for z in add_piece:
                            z.flag_conflict = True
                else:
                    # simply add and mark them all
                    add_piece = piece0 + piece1
                    for z in add_piece:
                        z.flag_conflict = True
                merged_toks.extend(add_piece)
                # add the hit one (linked)
                if pa is not None:
                    t_matched = to_merge_s0[pa]
                    t_matched.link(to_merge_s1[pb])
                    merged_toks.append(t_matched)
                lpa, lpb = pa, pb
            self.tokens = self.tokens[:-merge_k] + merged_toks + seg1.tokens[merge_k:]
        return

# the merger
class GreedyMerger:
    def __init__(self, args, bos, eos, mask):
        self.win_size: int = args.win_size  # each window's size
        self.merge_k: int = args.merge_k  # size considered for segment merging
        self.empty_score: float = float(np.log(args.empty_prob))  # default empty score
        self.skip_concat = args.skip_concat
        self.accu_score = args.accu_score
        self.bos, self.eos, self.mask = bos, eos, mask
        self.special_tok_set = {bos, eos, mask}
        self.delete_conflict, self.delete_gap, self.delete_link = args.delete_conflict, args.delete_gap, args.delete_link
        self.penalty_conflict, self.penalty_gap, self.penalty_link = args.penalty_conflict, args.penalty_gap, args.penalty_link
        self.fix_len_range, self.fix_len_firsteos_ratio = args.fix_len_range, args.fix_len_firsteos_ratio
        # about resolve_conflict in merging
        self.resolve_conflict, self.keep_conflict_merge = bool(args.resolve_conflict), bool(args.keep_conflict_merge)
        # --
        self.record_stat = args.record_stat
        self.stat = Counter()

    def summary(self):
        if self.record_stat:
            for k in sorted(self.stat.keys()):
                print(f"-- {k}: {self.stat[k]}")
        else:
            print("No stat available")

    # input is 2d array or 2d list, with optional scores for each token
    def run(self, windows: Union[List[List], np.ndarray], scores: Union[List[List], np.ndarray] = None,
            as_final_output=True, keep_rate=1., keep_rate_input_ratio=0., window_drop_rate=0., combine_repeat=True):
        _record_stat = self.record_stat
        stat = self.stat
        if scores is None:
            scores = [None] * len(windows)
        elif self.accu_score:  # accumulate scores inside the window
            for one_scores in scores:
                c = 0.
                for _i in range(len(one_scores)):
                    _oldc = c
                    c += one_scores[_i]
                    one_scores[_i] += _oldc
        # -----
        # drop low scored windows
        if window_drop_rate>0.:
            window_size = len(windows)
            window_scores = [np.mean(z) for z in scores]
            _K = min(int(window_size*window_drop_rate), window_size-1)
            drop_score_thresh = np.partition(window_scores, _K)[_K]
            drop_decisions = [z<drop_score_thresh for z in window_scores]
            drop_decisions[0] = False  # do not drop first and last window
            drop_decisions[-1] = False
        else:
            drop_decisions = [False] * len(windows)
        # -----
        # pass 0: deal with each window
        all_seqs: List[List[MToken]] = []
        for one_sidx, one_win in enumerate(windows):
            if drop_decisions[one_sidx]:
                continue  # drop this window!
            cur_seq: List[MToken] = []
            cur_scores = scores[one_sidx]
            if cur_scores is None:
                cur_scores = [0.] * len(one_win)  # by default 0.
            last_w = None
            for one_widx, one_w in enumerate(one_win):
                if combine_repeat and one_w == last_w:  # combine the two
                    cur_seq[-1].add((one_sidx, one_widx), cur_scores[one_widx])
                else:
                    cur_seq.append(MToken(one_w, (one_sidx, one_widx), cur_scores[one_widx]))
                last_w = one_w
            all_seqs.append(cur_seq)
        # -----
        # pass 1: concat
        if self.skip_concat:  # simply list all and solve all later
            segments = [MSeg(one_seq) for one_seq in all_seqs]
        else:  # first concat the continuous ones
            segments: List[MSeg] = []
            for one_seq in all_seqs:
                concat_ok = False  # by default failed concat
                if len(segments) > 0:
                    last_seg = segments[-1]
                    concat_ok = last_seg.concat(one_seq)
                if not concat_ok:  # if failed, add a new segment
                    segments.append(MSeg(one_seq))
        # -----
        # pass 2: solve conflicts and merge segments
        # todo(+N): there can be various kinds of orders, here we simply go from left to right
        seg0 = segments[0]
        for seg1 in segments[1:]:
            seg0.merge(seg1, self.merge_k, self.empty_score, self.resolve_conflict, self.keep_conflict_merge)
        # -----
        # pass 3: get final outputs
        # first delete all with flag_delete or speical tokens (bos/eos/mask)
        tokens: List[MToken] = []
        first_eos_id = None
        for t in seg0.tokens:
            if (not t.flag_delete) and (t.item not in self.special_tok_set):
                tokens.append(t)
            if first_eos_id is None and t.item == self.eos:  # record first eos position
                first_eos_id = min(t.ids, key=lambda x: sum(x))  # get the first position
        if first_eos_id is None:
            first_eos_id = (len(windows), 0)  # if no eos, then extend one
        # manually add bos and eos
        tokens = [MToken(self.bos, (0,0), 0.)] + tokens + [MToken(self.eos, first_eos_id, 0.)]
        first_eos_length = sum(first_eos_id)  # length according to first eos
        # record
        if _record_stat:
            stat["sent"] += 1
            stat[f"sent_outMin_{max(-10, min(len(tokens)-len(windows), 10))}"] += 1  # length mismatch: out-in
            stat["tok_in"] += len(windows)
            stat["tok_out"] += len(tokens)
            stat["tok_eoslen"] += first_eos_length
            for t in tokens:  # which type?
                stat[f"tok_[{t.flag_string()}]"] += 1
        # prepare output
        if as_final_output:
            ret_tokens = tokens  # as it is
        else:
            # =====
            # the simpler delete and mask method
            cur_output_len = len(tokens)
            # predefined deletion
            delete_flags = [((self.delete_conflict and z.flag_conflict) or (self.delete_gap and z.flag_gap)
                             or (self.delete_link and z.flag_link)) for z in tokens]
            token_scores = [z.get_score_with_penalties(self.penalty_conflict, self.penalty_gap, self.penalty_link)
                            for z in tokens]  # get scores with penalties for tokens
            # further to delete by score?
            _keep_rate_ref_len = (1.-keep_rate_input_ratio)*cur_output_len + keep_rate_input_ratio*len(windows)
            further_delete_budget = len(tokens) - sum(delete_flags) - int(keep_rate*_keep_rate_ref_len)
            if further_delete_budget > 0:
                for ii in np.argsort(token_scores):
                    if delete_flags[ii] or (tokens[ii].item in self.special_tok_set):  # already deleted or special
                        continue
                    delete_flags[ii] = True
                    further_delete_budget -= 1
                    if further_delete_budget<=0:
                        break
            # fix gap??
            _fix_len_range, _fix_len_eosratio = self.fix_len_range, self.fix_len_firsteos_ratio
            _target_len = int(_fix_len_eosratio*first_eos_length+(1-_fix_len_eosratio)*len(windows))  # set up target length
            if _fix_len_range < 1.:
                _target_range0, _target_range1 = int(_target_len*(1-_fix_len_range)), int(_target_len*(1+_fix_len_range))
            else:
                assert _fix_len_range>=1.
                _target_range0, _target_range1 = int(_target_len-_fix_len_range), int(_target_len+_fix_len_range)
            if cur_output_len < _target_range0:  # need to add
                cur_len_gap = cur_output_len - _target_range0
            elif cur_output_len > _target_range1:  # need to delete
                cur_len_gap = cur_output_len - _target_range1
            else:
                cur_len_gap = 0  # length is ok
            if cur_len_gap != 0:
                # filter out deleted ones and get gaps
                filtered_tokens = []  # kept tokens
                filtered_avg_positions = []  # avg positions of the kept tokens
                filtered_num_insertions = []  # how many masks already there (after each one)
                filtered_gaps = []  # current gap (how many masks still needed) according to posi-diff and num-mask (after each one)
                for one_tok, one_dflag in zip(tokens, delete_flags):
                    if one_dflag:
                        filtered_num_insertions[-1] += 1  # must be there since bos cannot be deleted
                    else:
                        # put the gap
                        cur_avg_posi = np.mean([sum(ii) for ii in one_tok.ids])
                        if len(filtered_avg_positions)>0:
                            # how many masks still needed
                            filtered_gaps.append(cur_avg_posi-filtered_avg_positions[-1]-1-filtered_num_insertions[-1])
                        # make a new one
                        filtered_tokens.append(one_tok)
                        filtered_avg_positions.append(cur_avg_posi)
                        filtered_num_insertions.append(0)
                filtered_gaps.append(0.)  # for the last one
                assert len(filtered_gaps) == len(filtered_tokens)
                # adjust according to length
                gap_budget = cur_len_gap
                if gap_budget < 0:  # add new masks
                    gap_heap = [(-x,i) for i,x in enumerate(filtered_gaps)]  # min-heap
                    heapq.heapify(gap_heap)  # min-heap
                    while gap_budget < 0:
                        _g, _i = heapq.heappop(gap_heap)  # next one
                        filtered_num_insertions[_i] += 1  # add one mask
                        heapq.heappush(gap_heap, (_g+1, _i))  # add one since this is min-heap
                        gap_budget += 1
                else:  # try to delete existing masks
                    gap_heap = [(x,i) for i,x in enumerate(filtered_gaps)]  # min-heap
                    heapq.heapify(gap_heap)  # min-heap
                    while gap_budget > 0:
                        _g, _i = heapq.heappop(gap_heap)  # next one
                        filtered_num_insertions[_i] -= 1  # minus one mask, notice that this can become negative!
                        heapq.heappush(gap_heap, (_g+1, _i))
                        gap_budget -= 1
                # finally get final output
                # insert masks
                mask_tok = MToken(self.mask, None, 0.)  # padding mask token
                ret_tokens = []
                for one_tok, one_ins_num in zip(filtered_tokens, filtered_num_insertions):
                    ret_tokens.append(one_tok)
                    ret_tokens.extend([mask_tok] * one_ins_num)  # [M]*NEG_NUM is still empty []
            else:
                # simply no need to add or delete masks
                mask_tok = MToken(self.mask, None, 0.)  # padding mask token
                ret_tokens = [(mask_tok if one_dflag else one_tok) for one_tok, one_dflag in zip(tokens, delete_flags)]
            if _record_stat:
                stat[f"Msent_outMin_{max(-10, min(len(ret_tokens) - len(windows), 10))}"] += 1  # length mismatch: out-in
                stat[f"Msent_gap_{max(-10, min(cur_len_gap, 10))}"] += 1
                stat["Mtok_out"] += len(ret_tokens)
                stat["Mtok_dflag"] += sum(delete_flags)
        # -----
        ret_tokens = [r.item for r in ret_tokens]
        return ret_tokens, tokens  # target_tokens, orig_tokens

# =====

# readers
def reader_line(fin, win_size: int):
    for line in fin:
        raw_seq_list = line.split()
        seq_list = [raw_seq_list[i:i+win_size] for i in range(0, len(raw_seq_list), win_size)]
        yield {"windows": seq_list}

def reader_json(fin, *args):
    for line in fin:
        yield json.loads(line)

def reader_special(fin, *args):
    line = fin.readline()  # start
    while True:
        if len(line) == 0:
            break
        assert line.startswith("ref:")
        extra_info = {"ref": line, "len_ref": len(line.split())-1}
        seq_list = []
        while True:  # read hyp
            line = fin.readline()
            if len(line) == 0 or not line.startswith("hyp:"):
                break
            seq_list.append(line[4:].split())
        yield {"windows": seq_list, "extra_info": extra_info}

# small test function
def count_bigrams(file):
    with open(file) as fd:
        all_count = 0
        same_count = 0
        for line in fd:
            tokens = line.split()
            for a,b in zip(tokens, tokens[1:]):
                all_count += 1
                same_count += int(a==b)
        print(f"{same_count}/{all_count}={same_count/all_count}")

# from typing import List, Union, Set
def tokens2str(tokens: List[str], do_postprocess: bool, delete_set: Set):
    if do_postprocess:
        final_strs = []
        for s in tokens:
            if s not in delete_set:
                if len(final_strs)>0 and final_strs[-1].endswith("@@"):
                    final_strs[-1] = final_strs[-1][:-2] + s
                else:
                    final_strs.append(s)
    else:
        final_strs = tokens
    return " ".join(final_strs)

def token2str_file(in_file, out_file, do_postprocess: bool, delete_set: Set):
    with open(in_file) as rfd, open(out_file, 'w') as wfd:
        for line in rfd:
            tokens = line.split()
            ss = tokens2str(tokens, do_postprocess, delete_set)
            wfd.write(ss+"\n")

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='')
    parser.add_argument("-o", "--output", type=str, default='')
    parser.add_argument("-f", "--format", type=str, default="line", choices=["line", "json", "special"])
    parser.add_argument("--verbose", action='store_true', help="Whether print more info")
    # --
    parser.add_argument("--as_final_output", action='store_true', help="Whether as final layer output")
    parser.add_argument("--keep_rate", type=float, default=1., help="How much tokens to keep for the current iter")
    parser.add_argument("--do_postprocess", action='store_true', help="Whether to do post-processing")
    # --
    parser.add_argument("--record_stat", type=int, default=0)
    parser.add_argument("--win_size", type=int, default=3, help="Window size")
    parser.add_argument("--merge_k", type=int, default=3, help="How many s0[-K:] and s1[:K] to merge?")
    parser.add_argument("--skip_concat", type=int, default=0, help="Whether skip concat pass and direct go to merge pass")
    parser.add_argument("--empty_prob", type=float, default=0.25,
                        help="Default prob (score=logprob) for empty seq when resolving conflicts")
    parser.add_argument("--accu_score", type=int, default=0, help="Accumulate score inside one window")
    parser.add_argument("--resolve_conflict", type=int, default=1, help="Whether resolve conflicts in merging?")
    parser.add_argument("--keep_conflict_merge", type=int, default=0, help="Keep conflicts in merging and delete later?")
    parser.add_argument("--mask", type=str, default="[M]")
    parser.add_argument("--bos", type=str, default="<s>")
    parser.add_argument("--eos", type=str, default="</s>")
    parser.add_argument("--delete_conflict", type=int, default=0)  # only for Non-as_final_output mode, same for CGL
    parser.add_argument("--delete_gap", type=int, default=0)
    parser.add_argument("--delete_link", type=int, default=0)
    parser.add_argument("--penalty_conflict", type=float, default=0.)  # only for Non-as_final_output mode, same for CGL
    parser.add_argument("--penalty_gap", type=float, default=0.)
    parser.add_argument("--penalty_link", type=float, default=0.)
    parser.add_argument("--fix_len_range", type=float, default=0.1, help="Fix length by adding or deleting masks: if <1., then towards [LEN*(1-this), LEN*(1+this)], if >=1, [LEN-this, LEN+this]")
    parser.add_argument("--fix_len_firsteos_ratio", type=float, default=0., help="For fixing length, which length to use: input window number or first eos position?")
    args = parser.parse_args()
    # -----
    fin = sys.stdin if (args.input == "-" or args.input == "") else open(args.input)
    fout = sys.stdout if (args.output == "-" or args.output == "") else open(args.output, "w")
    reader = {"line": reader_line, "json": reader_json, "special": reader_special}[args.format](fin, args.win_size)
    merger = GreedyMerger(args, args.bos, args.eos, args.mask)
    len_ref, len_in, len_out = 0, 0, 0
    delete_set = {args.bos, args.eos, args.mask}
    for one_inst in reader:
        seq_list = one_inst["windows"]
        extra_info = one_inst.get("extra_info", {})
        ret_tokens, orig_tokens = merger.run(seq_list, scores=one_inst.get("scores"),
                                             as_final_output=args.as_final_output, keep_rate=args.keep_rate)
        if args.verbose:
            fout.write(f"# -- \n")
            for one_sidx, one_seq in enumerate(seq_list):
                fout.write(f"W{one_sidx}: " + " ".join(one_seq) + "\n")
            fout.write(str(orig_tokens)+"\n")
            fout.write(str(ret_tokens)+"\n")
            fout.write(tokens2str([z.item for z in ret_tokens], args.do_postprocess, delete_set)+"\n")
            fout.write(f"-- {extra_info}\n")
        else:
            fout.write(tokens2str([z.item for z in ret_tokens], args.do_postprocess, delete_set)+"\n")
        len_ref += extra_info.get("len_ref", 0)
        len_in += len(seq_list)
        len_out += len(ret_tokens)
    if args.verbose:
        print(f"{len_out}/{len_ref+1e-5}={len_out/(len_ref+1e-5)}")
        print(f"{len_out}/{len_in+1e-5}={len_out/(len_in+1e-5)}")
        merger.summary()
    fin.close()
    fout.close()

if __name__ == '__main__':
    main()

"""
# example input 
<s> unfortunately , unfortunately , it , it &apos;s , this is &apos;s the good the good news good news , news , because news , because , because there because there &apos;s are some other some other problems other problems , problems , and problems and they they &apos;ve often often mentioned before mentioned before . mentioned . </s>
"""
# run
# python3 nat_merge.py -i translation_rnn0501.json -o _tmpv --format json --record_stat 1 --verbose --keep_rate 0.5
# python3 nat_merge.py -i translation_rnn0501.json -o _tmp1 --format json --as_final_output --do_postprocess
