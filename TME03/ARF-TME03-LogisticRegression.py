#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arftools import *

def sigmoid(w,x):
    return 1./(1.+np.exp(x.dot(w)))
def grad_sigmoid(w,x,y):
    return (sigmoid(w,x)-y).dot(x)

class LogisticRegression(Classifier,OptimFunc,GradientDescent):
    def __init__(self,eps=1e-4,max_iter=5000,delta=1e-6,dim=2):
        self.dim = dim
        GradientDescent.__init__(self,self,eps,max_iter,delta)
    def fit(self,data,y):
        self.data = data
        self.y    = y
        self.optimize()
    def f(self,w):
        return sum(sigmoid(w,self.data))
    def grad_f(self,w):
        return grad_sigmoid(w,self.data,self.y)
    def predict(self,testX):
        X = to_array(testX)
        return -np.sign(sigmoid(self.x,X) - 0.5)

trainX,trainY = gen_arti()
testX ,testY  = gen_arti()
lr = LogisticRegression()
lr.fit(trainX,trainY)
# print lr.predict(testX)
print lr.score(testX,testY)

plot(testX,testY,lr.predict,step=100)



