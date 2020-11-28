#!/usr/bin/python
# -*- coding: utf-8 -*-

# USAGE: python fusion_avg.py,
# used for fusion of 2 or 3 streams

import os
import numpy as np
import argparse
import pdb

stream_dir_header = 'veriset/normalized-scores/stream'
fusion_dir_header = 'veriset/'

stream1_weight = 0.05
stream2_weight = 0.53
stream3_weight = 0.42


# ==================== === ====================
def GetArgs():
    parser = argparse.ArgumentParser(description="Fusion for multi-stream")
    parser.add_argument('--stream_num', type=int, default=3, help='number of streams for fusion, max 3')
    parser.add_argument('--scorefile_1', type=str, default=stream_dir_header + '1.txt',
                        help='original scores file from stream 1')
    parser.add_argument('--scorefile_2', type=str, default=stream_dir_header + '2.txt',
                        help='original scores file from stream 2')
    parser.add_argument('--scorefile_3', type=str, default=stream_dir_header + '3.txt',
                        help='original scores file from stream 3')
    parser.add_argument('--fused_scorefile', type=str, default=fusion_dir_header + 'scores.txt',
                        help='Fused scores file')

    args = parser.parse_args()
    return args


# ==================== === ====================

def score_fuse(streams, original_scores):
    scores = []  # initialization
    for index in range(len(original_scores)):
        if streams == 2:
            score = (stream1_weight * original_scores[index][0] +
                     stream2_weight * original_scores[index][1])
        elif streams == 3:
            score = (original_scores[index][0] * stream1_weight +
                     original_scores[index][1] * stream2_weight +
                     original_scores[index][2] * stream3_weight)
        # limit the range to [0.0, 1.0]
        score = min(score, 1.0)
        score = max(score, 0.0)
        scores.append(score)
    return scores


# ==================== === ====================

def read_file(filename):
    scores_file = open(filename, 'r').readlines()
    scores = []
    utt1s = []
    utt2s = []
    # you may also want to remove whitespace characters like `\n` at the end of each line
    for line in scores_file:
        score, utt1, utt2 = line.rstrip().split()
        scores.append(float(score))
        utt1s.append(utt1)
        utt2s.append(utt2)
    return scores, utt1s, utt2s


# ==================== === ====================
def main():
    args = GetArgs()
    if args.stream_num <= 1:
        raise ValueError("Streams is smaller than allowed, stop!!!")
    elif args.stream_num > 3:
        raise ValueError("Streams is larger than allowed, stop!!!")

    # Step1: get original scores
    scores1, utt11, utt12 = read_file(args.scorefile_1)
    scores2, utt21, utt22 = read_file(args.scorefile_2)
    if args.stream_num >= 3:
        scores3, utt31, utt32 = read_file(args.scorefile_3)
        scores_group = list(zip(scores1, scores2, scores3))
    else:
        scores_group = list(zip(scores1, scores2))

    # Step2: score averaging
    fused_scores = score_fuse(args.stream_num, scores_group)

    # Step3: save scores
    with open(args.fused_scorefile, 'w') as outfile:
        for index in range(len(fused_scores)):
            outfile.write('%.10f %s %s\n' % (fused_scores[index], utt11[index], utt12[index]))

    # print statistics
    # print('minimum distance: %.6f, maximum distance: %.6f' % (distance_min, distance_max))
    quit()


if __name__ == "__main__":
    main()
