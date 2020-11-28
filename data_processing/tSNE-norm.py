#!/usr/bin/python
# -*- coding: utf-8 -*-

# USAGE: python tSNE-norm.py,
# used to do tSNE normalization


import argparse
import numpy as np
import matplotlib.pyplot as plt
target_set = 'veriset/'
target_txt = 'stream1.txt'
original_score_file = target_set + 'original-scores/' + target_txt
normalized_score_file = target_set + 'normalized-scores/' + target_txt


# ==================== === ====================
def GetArgs():
    parser = argparse.ArgumentParser(description="VoxSRC")
    parser.add_argument('--original_scores', type=str, default=original_score_file,
                        help='original scores file from trainer')
    parser.add_argument('--normalized_scores', type=str, default=normalized_score_file,
                        help='Normalized scores file, within range of [0, 1]')
    parser.add_argument('--positive', type=int, default=1,
                        help='1 if higher is positive; 0 if lower is positive')

    args = parser.parse_args()
    return args


# ==================== === ====================

def score_norm(y, pos):
    # y denotes original scores or rather distance
    # get statistics, min, max, mean, variance
    y_min = min(y)
    y_max = max(y)
    y_mean = np.mean(y)
    y_var = np.var(y)
    print('original scores range: [%.4f, %.4f]; mean: %.6f; variance: %.6f' % (y_min, y_max, y_mean, y_var))

    # plot the distribution on original scores
    '''
    list.sort(y)  # sort new list
    x = np.arange(0, len(y))
    plt.scatter(x, y, c=y, cmap='Reds')
    plt.xlabel('index')
    plt.ylabel('distance')
    plt.show()
    '''

    # get the distribution of original scores
    '''
    counter = np.zeros(6)
    for index in range(len(y)):
        if y[index] < -1.25:
            counter[0] += 1
        elif y[index] < -1.1:
            counter[1] += 1
        elif y[index] < y_mean:
            counter[2] += 1
        elif y[index] < -0.9:
            counter[3] += 1
        elif y[index] < -0.75:
            counter[4] += 1
        else:
            counter[5] += 1
    plt.figure(2)  # https://www.cnblogs.com/calvin-zhang/articles/10272300.html
    region = np.arange(0, len(counter))
    plt.scatter(region, counter, c='r', cmap='Reds')
    plt.xlabel('region')
    plt.ylabel('number')
    plt.show()
    '''

    # score normalization, input score is negative Euclidean distance
    d_thre = y_mean
    neg_d_max = d_thre + 0.25  # y_mean + 0.25
    neg_d_min = d_thre - 0.25  # y_mean - 0.25
    y_score = y  # initialization
    for index in range(len(y)):
        if pos == 1:
            # pos is 1, means higher is positive
            if y[index] > y_max:
                y_score[index] = 1.0
            elif y[index] < y_min:
                y_score[index] = 0.0
            else:
                y_score[index] = np.divide(1.0, 1 + y[index] * y[index])
        else:
            # pos is 0, means lower is positive
            raise ValueError('position is not supported!')
        # limit the range to [0.0, 1.0]
        y_score[index] = min(y_score[index], 1.0)
        y_score[index] = max(y_score[index], 0.0)
    return y_score, y_mean, y_var


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

    # Step1: get original scores
    y, utt1, utt2 = read_file(args.original_scores)

    # Step2: score normalization
    y_score, mean, variance = score_norm(y, args.positive)

    # Step3: save scores
    with open(args.normalized_scores, 'w') as outfile:
        for index in range(len(y_score)):
            outfile.write('%.10f %s %s\n' % (y_score[index], utt1[index], utt2[index]))
    # quit()

    # print statistics
    print('mean: %.6f, variance: %.6f' % (mean, variance))


if __name__ == "__main__":
    main()
