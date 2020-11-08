#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.softmax as softmax
import loss.angleproto as angleproto

from accuracy import accuracy

class LossFunction(nn.Module):

    def __init__(self, nOut, nClasses, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.LossFunction(nOut=nOut, nClasses=nClasses)
        self.angleproto = angleproto.LossFunction()

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.view(-1,x.size()[-1]), label.repeat_interleave(2))

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1