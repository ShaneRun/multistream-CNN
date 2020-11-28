#!/usr/bin/python
# -*- coding: utf-8 -*-

import os


# ==================== === ====================
def main():
    # EER

    os.system("python ./compute_EER.py --prediction veriset/scores.txt "
              "--ground_truth veriset/trials.txt")
    """
    os.system("python ./compute_EER.py --prediction veriset/normalized-scores/fus1_1.txt "
              "--ground_truth veriset/trials.txt")
    os.system("python ./compute_EER.py --prediction veriset/normalized-scores/fus1_2.txt "
              "--ground_truth veriset/trials.txt")
    os.system("python ./compute_EER.py --prediction veriset/normalized-scores/fus1_3.txt "
              "--ground_truth veriset/trials.txt")
    os.system("python ./compute_EER.py --prediction veriset/normalized-scores/fus1_4.txt "
              "--ground_truth veriset/trials.txt")
    """
    # minDCF

    os.system("python ./compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 --scores-filename "
              "veriset/scores.txt --trials-filename veriset/trials.txt")
    """
    os.system("python ./compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 --scores-filename "
              "veriset/normalized-scores/fus1_1.txt --trials-filename veriset/trials.txt")
    os.system("python ./compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 --scores-filename "
              "veriset/normalized-scores/fus1_2.txt --trials-filename veriset/trials.txt")
    os.system("python ./compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 --scores-filename "
              "veriset/normalized-scores/fus1_3.txt --trials-filename veriset/trials.txt")
    os.system("python ./compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 --scores-filename "
              "veriset/normalized-scores/fus1_4.txt --trials-filename veriset/trials.txt")
    """
    quit()


if __name__ == "__main__":
    main()
