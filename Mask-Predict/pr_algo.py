#

from typing import List
import sys
import json
import numpy as np
from fairseq import pybleu


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence

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
# special routine to merge a series of sequences (incrementally, left to right)
# input: seq_list is List[List[item]], K is merging range in each step
def merge_seqs(seq_list: List[List], K: int):
    cur_seq = []
    for s in seq_list:
        if len(cur_seq) < K:
            cur0, cur1 = [], cur_seq
        else:
            cur0, cur1 = cur_seq[:-K], cur_seq[-K:]  # use the most recent K for merging
        align_res = align_seqs(cur1, s)
        # get the merged one
        ma1, ma2 = align_res[:2]
        cur2 = [(s[b] if a is None else cur1[a]) for a,b in zip(ma1, ma2)]
        # finally concat
        cur_seq = cur0 + cur2
    return cur_seq

# ==
def test1():
    # test
    s1 = "pero las horas son fijadas por ' provincia ' , no por el gobierno central ...".split()
    s2 = "pero las horas se establecen por provincia , no por ' gobierno central ' ...".split()
    rets = align_seqs(s1, s2)
    s_merge = [f"{s1[a] if a else ''}/{s2[b] if b else ''}" for a, b in zip(rets[0], rets[1])]
    # breakpoint()
    print(rets)

def test2():
    SLEN = 20
    PLEN = 10
    WIN = 2
    original_seq = list(range(SLEN))
    center_seq = sorted(np.random.randint(0, SLEN, size=PLEN))
    pieces = [(list(range(r-WIN, r)) + [r] + list(range(r+1, r+WIN+1))) for r in center_seq]
    ret_seq = merge_seqs(pieces, K=5)
    
    print(ret_seq, center_seq)

def main(filename, SEG=3, K=3):
    scorer = pybleu.PyBleuScorer()
    IGNORE_SET = ["<s>", "</s>"]
    SEG = int(SEG)  # SEG is used when reading raw inputs
    K = int(K)  # K is tunable for merging
    pred_sens = []
    cur_ref = ''
    cur_hyp = ''

    with open(filename) as f:
        for line in f:
            if line.startswith('ref:'):
                if len(cur_ref) > 0:
                    pred_sens.append((cur_ref, cur_hyp))
                    
                    cur_hyp = ''
                cur_ref = line.split(':', 1)[1].strip()
            elif line.startswith('hyp:'):
                cur_hyp = ' '.join((cur_hyp, line.split(':', 1)[1].strip()))
    results = []
    with open('pred.txt', 'w') as predf, open('ref.txt', 'w') as reff:
        for line in pred_sens:
            ref, pred = line

            try:
                seq_list = json.loads(line)  # each line is a json of List[List]
            except:
                raw_seq_list = pred.split()
                seq_list = [raw_seq_list[i:i+SEG] for i in range(0, len(raw_seq_list), SEG)]
            ret_seq = merge_seqs(seq_list, K)
            final_seq = [""]
            for r in ret_seq:
                if r != final_seq[-1] and r not in IGNORE_SET:
                    final_seq.append(r)
            ori_ref = process_bpe_symbol(ref.strip(), '@@ ')
            reff.write(ori_ref + '\n')
            ori_pred = process_bpe_symbol(' '.join(final_seq).strip(), '@@ ')
            results.append((ori_ref, ori_pred))
            predf.write(ori_pred + '\n')
    ref, out = zip(*results)
    print(scorer.score(ref, out))
        

if __name__ == '__main__':
    # test1()
    # test2()
    main(*sys.argv[1:])
