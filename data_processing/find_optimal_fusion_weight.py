#!/usr/bin/python
# -*- coding: utf-8 -*-

# USAGE: python find_optimal_fusion_weight_old.py,
# used to find out the optimal fusion weight
# 2/3 streams

from __future__ import print_function

import argparse
import time
from operator import itemgetter

import numpy as np

stream_dir_header = 'veriset/normalized-scores/stream'
fusion_dir_header = 'veriset/'
p_target = 0.05
c_miss = 1.0
c_fa = 1.0


# ==================== === ====================
def GetArgs():
    parser = argparse.ArgumentParser(description="Fusion for multi-stream")
    parser.add_argument('--stream_num', type=int, default=3, help='number of stream for fusion, max 3')
    parser.add_argument('--scorefile_1', type=str, default=stream_dir_header + '1.txt',
                        help='original scores file from stream 1')
    parser.add_argument('--scorefile_2', type=str, default=stream_dir_header + '2.txt',
                        help='original scores file from stream 2')
    parser.add_argument('--scorefile_3', type=str, default=stream_dir_header + '3.txt',
                        help='original scores file from stream 3')
    parser.add_argument('--trialfile', type=str, default=fusion_dir_header + 'trials.txt',
                        help='trial file storing ground truth')
    parser.add_argument('--fused_scorefile', type=str, default=fusion_dir_header + 'scores.txt',
                        help='Fused scores file')

    args = parser.parse_args()
    return args


# ==================== === ====================

def read_score(filename):
    scores_file = open(filename, 'r').readlines()
    scores = []
    # you may also want to remove whitespace characters like `\n` at the end of each line
    for line in scores_file:
        score, utt1, utt2 = line.rstrip().split()
        scores.append(float(score))
    return scores


# ==================== === ====================

def score_fuse(streams, original_scores, weight):
    scores = []  # initialization
    for index in range(len(original_scores)):
        score = 0
        for stream_index in range(streams):
            score += original_scores[index][stream_index] * weight[stream_index]
        # limit the range to [0.0, 1.0]
        score = min(score, 1.0)
        score = max(score, 0.0)
        scores.append(score)
    return scores


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


# ==================== === ====================
def main():
    args = GetArgs()
    if args.stream_num <= 1:
        raise ValueError("Streams is smaller than allowed, stop!!!")
    elif args.stream_num > 3:
        raise ValueError("Streams is larger than allowed, stop!!!")

    # Step 1: initialization
    it_counter = 0
    wt = np.zeros(3, float)

    # Step 2: constraint
    wt_max = 1.0
    wt_min = 0.0
    wt_it_step = 0.01

    # Step 3: resolving for optimal weight
    # 3.1 read original scores and trials
    scores1 = read_score(args.scorefile_1)
    scores2 = read_score(args.scorefile_2)
    if args.stream_num >= 3:
        scores3 = read_score(args.scorefile_3)
        scores_group = list(zip(scores1, scores2, scores3))
    else:
        scores_group = list(zip(scores1, scores2))

    labels = read_score(args.trialfile)

    # 3.2 iteration control: update of weight
    # first condition: weight of stream 1, decrease from wt_max to wt_min
    # second condition: weight of stream 2
    wt[0] = wt_max
    global_opt = [1.0, 0.0, wt[0], wt[1], wt[2]]

    while wt[0] >= wt_min:
        local_opt = [1.0, 0.0, wt[0], wt[1], wt[2]]
        wt[1] = wt_max - wt[0]
        while wt[1] >= wt_min:
            # calculate weight
            wt[2] = wt_max - wt[1] - wt[0]
            # score fusion
            scores = score_fuse(args.stream_num, scores_group, wt)
            # compute minDCF
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

            # update of local optimal weight
            if mindcf < local_opt[0]:
                local_opt[0] = mindcf
                local_opt[1] = threshold
                local_opt[2] = wt[0]
                local_opt[3] = wt[1]
                local_opt[4] = wt[2]
            if args.stream_num >= 3:
                # go to next loop
                wt[1] -= wt_it_step
            else:
                # skip current loop
                wt[1] = -wt_it_step
        # print local optimal results
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it_counter,
              "local minDCF is %.4f, with threshold %.4f" % (local_opt[0], local_opt[1]),
              "weight: %.3f, %.3f, %.3f" % (local_opt[2], local_opt[3], local_opt[4]))
        # update of global optimal weight
        if local_opt[0] < global_opt[0]:
            global_opt = local_opt
        it_counter += 1
        wt[0] -= wt_it_step

    # Step 4: output optimal result
    # print global optimal results
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[Searching finished]")
    print("global minDCF is %.4f, with threshold %.4f" % (global_opt[0], global_opt[1]),
          "weight: %.3f, %.3f, %.3f" % (global_opt[2], global_opt[3], global_opt[4]))
    quit()


if __name__ == "__main__":
    main()
