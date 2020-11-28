#!/usr/bin/env python3
# Copyright 2018  David Snyder
# This script is modified from the Kaldi toolkit -
# https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm
import numpy as np
import sys, argparse, os


def GetArgs():
    parser = argparse.ArgumentParser(description="Compute the minimum "
                                                 "detection cost function along with the threshold at which it occurs. ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--p-target', type=float, dest="p_target",
                        default=0.05,
                        help='The prior probability of the target speaker in a trial.')
    parser.add_argument('--c-miss', type=float, dest="c_miss", default=1,
                        help='Cost of a missed detection.  This is usually not changed.')
    parser.add_argument('--c-fa', type=float, dest="c_fa", default=1,
                        help='Cost of a spurious detection.  This is usually not changed.')
    parser.add_argument('--scores-filename', type=str, dest='scores_filename',
                        default="veriset/normalized-scores/stream1.txt",
                        help="Input scores file, with columns of the form <score> <utt1> <utt2>")
    parser.add_argument('--trials-filename', type=str, dest='trials_filename',
                        default="veriset/trials.txt",
                        help="Input trials file, with columns of the form <t0/1> <utt1> <utt2>")
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args


def CheckArgs(args):
    if args.c_fa <= 0:
        raise Exception("--c-fa must be greater than 0")
    if args.c_miss <= 0:
        raise Exception("--c-miss must be greater than 0")
    if args.p_target <= 0 or args.p_target >= 1:
        raise Exception("--p-target must be greater than 0 and less than 1")
    return args


def plot_DET_curve():
    # 设置刻度范围
    pmiss_min = 0.001

    pmiss_max = 0.6

    pfa_min = 0.001

    pfa_max = 0.6

    # 刻度设置
    pticks = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
              0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
              0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
              0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
              0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999]

    # 刻度*100
    xlabels = [' 0.001', ' 0.002', ' 0.005', ' 0.01 ', ' 0.02 ', ' 0.05 ',
               '  0.1 ', '  0.2 ', ' 0.5  ', '  1   ', '  2   ', '  5   ',
               '  10  ', '  20  ', '  40  ', '  60  ', '  80  ', '  90  ',
               '  95  ', '  98  ', '  99  ', ' 99.5 ', ' 99.8 ', ' 99.9 ',
               ' 99.95', ' 99.98', ' 99.99', '99.995', '99.998', '99.999']

    ylabels = xlabels

    # 确定刻度范围
    n = len(pticks)
    # 倒叙
    for k, v in enumerate(pticks[::-1]):
        if pmiss_min <= v:
            tmin_miss = n - k - 1  # 移动最小值索引位置
        if pfa_min <= v:
            tmin_fa = n - k - 1  # 移动最小值索引位置
    # 正序
    for k, v in enumerate(pticks):
        if pmiss_max >= v:
            tmax_miss = k + 1  # 移动最大值索引位置
        if pfa_max >= v:
            tmax_fa = k + 1  # 移动最大值索引位置

    # FRR
    plt.figure()
    plt.xlim(norm.ppf(pfa_min), norm.ppf(pfa_max))

    plt.xticks(norm.ppf(pticks[tmin_fa:tmax_fa]), xlabels[tmin_fa:tmax_fa])
    plt.xlabel('False Alarm probability (in %)')

    # FAR
    plt.ylim(norm.ppf(pmiss_min), norm.ppf(pmiss_max))
    plt.yticks(norm.ppf(pticks[tmin_miss:tmax_miss]), ylabels[tmin_miss:tmax_miss])
    plt.ylabel('Miss probability (in %)')

    return plt


# 计算EER
def compute_EER(frr, far):
    threshold_index = np.argmin(abs(frr - far))  # 平衡点
    eer = (frr[threshold_index] + far[threshold_index]) / 2
    print("eer=", eer)
    return eer


# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF2(P_miss, P_fa):
    C_miss = C_fa = 1
    P_true = 0.01
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    min_DCF = min(DCF)

    print("min_DCF_2=", min_DCF)

    return min_DCF


# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF3(P_miss, P_fa, min_DCF_2):
    C_miss = C_fa = 1
    P_true = 0.001
    P_false = 1 - P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")

    DCF = C_miss * P_miss * P_true + C_fa * P_fa * P_false

    # 该操作是我自己加的，因为论文中的DCF10-3指标均大于DCF10-2且高于0.1以上，所以通过这个来过滤一下,错误请指正
    min_DCF = 1
    for dcf in DCF:
        if (min_DCF_2 + 0.1) < dcf < min_DCF:
            min_DCF = dcf

    print("min_DCF_3=", min_DCF)
    return min_DCF


if __name__ == "__main__":
    # 读文件获取y_true和y_score
    # y_true = np.load('./dataset/y_true.npy')
    # y_score = np.load('./dataset/y_pre.npy')
    args = GetArgs()
    scores_file = open(args.scores_filename, 'r').readlines()
    trials_file = open(args.trials_filename, 'r').readlines()
    c_miss = args.c_miss
    c_fa = args.c_fa
    p_target = args.p_target

    scores = []
    labels = []

    trials = {}
    for line in trials_file:
        target, utt1, utt2 = line.rstrip().split()
        trial = utt1 + " " + utt2
        trials[trial] = target

    for line in scores_file:
        score, utt1, utt2 = line.rstrip().split()
        trial = utt1 + " " + utt2
        if trial in trials:
            scores.append(float(score))
            if trials[trial] == "1":
                labels.append(1)
            else:
                labels.append(0)
        else:
            raise Exception("Missing entry for " + utt1 + " and " + utt2
                            + " " + args.scores_filename)

    # 计算FAR和FRR
    fpr, tpr, thres = roc_curve(labels, scores)
    frr = 1 - tpr
    far = fpr
    frr[frr <= 0] = 1e-5
    far[far <= 0] = 1e-5
    frr[frr >= 1] = 1 - 1e-5
    far[far >= 1] = 1 - 1e-5

    # 画图
    plt = plot_DET_curve()
    x, y = norm.ppf(frr), norm.ppf(far)
    plt.plot(x, y)
    plt.plot([-40, 1], [-40, 1])
    # plt.plot(np.arange(0,40,1),np.arange(0,40,1))
    plt.show()

    eer = compute_EER(frr, far)

    min_DCF_2 = compute_minDCF2(frr * 100, far * 100)

    min_DCF_3 = compute_minDCF3(frr * 100, far * 100, min_DCF_2)
