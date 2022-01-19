# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Descente de gradient et Perceptron
# 
# ## Intro et reprise du code de la semaine précédente
# 
# Télécharger sur le site de l'ue le script arftools.py.
# 
# Etudier le code suivant, repris de la semaine dernière, modifié afin de faciliter l'utilisation : OptimFunc est maintenant une classe que l'on instancie avec la définition de la fonction, du gradient et de la dimension.

# <codecell>

import numpy as np
from numpy import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from arftools import *

def xcosx_f(x):
    return x*np.cos(x)
def xcosx_grad(x):
    return np.cos(x)-x*np.sin(x)

xcosx=OptimFunc(xcosx_f,xcosx_grad,1)
grd = GradientDescent(xcosx,max_iter=5000)
# grd.optimize()

# <markdowncell>

# Adapter votre code pour Rosenbrock. Etudier également la fonction d'erreur quadratique : f(x)=x1^2+alpha*x2^2
class Rosenbrock(OptimFunc):
    def __init__(self,alpha):
        self.dim   = 2
        self.alpha = alpha

    def f(self,x):
        x   = to_array(x)
        x_1 = x[:,0]
        x_2 = x[:,1]
        return np.array(x_1**2 + self.alpha*(x_2**2))

    def grad_f(self,x):
        x = to_array(x)
        x_1 = x[:,0]
        x_2 = x[:,1]
        return np.array(2*x_1+2*alpha*x_2)

# ## Perceptron
# 
# + Implémenter les fonctions hinge_f(data,y,w) et hinge_grad(data,y,w) qui représentent l'erreur du perceptron pour les données data, les labels y (-1 ou 1), et le vecteur de poids w; vous utiliserez np.maximum qui renvoie le maximum terme à terme et np.sign qui renvoie le signe (attention, vous aurez à utiliser le fait que np.sign(0)=0). Faites bien attention aux dimensions de chaque terme que vous utiliserez. 
def hinge_f(data,y,w): # x: n*d, y: n, w: (1,d)
    return np.maximum(-data.dot(w.T)*y,0)

def hinge_grad(data,y,w): # return (n,d)
    return ((np.sign(hinge_f(data,y,w)).T*(-y))*(data.T)).T
    
# + Tester sur des données artificielles.
# 
# + Coder une classe Perceptron qui hérite de Classifier, OptimFunc, et GradientDescent.  Tester la sur des données artificielles.
class Perceptron(Classifier,OptimFunc,GradientDescent):
    def __init__(self,eps=1e-4,max_iter=5000,delta=1e-6): # ,dim=2):
        # self.dim = dim
        GradientDescent.__init__(self,self,eps,max_iter,delta)
    def fit(self,data,y):
        self.dim  = len(data[0])
        self.data = data
        self.y    = y
        self._reset= True
        self.optimize(self._reset)
    def f(self,w):
        return hinge_f(self.data,self.y,w).sum()
    def grad_f(self,w):
        return hinge_grad(self.data,self.y,w).sum(0)
    def predict(self,testX):
        X = to_array(testX)
        return np.sign(X.dot(self.x))
    
def quadratique_f(data,y,w):
    return ((y - data.dot(w.T))**2)/2
    
def quadratique_grad(data,y,w):
    return -(y - data.dot(w.T)).dot(data)

class PerceptronQuad(Perceptron):
    def f(self,w):
        return quadratique_f(self.data,self.y,w).sum()/len(self.data)
    def grad_f(self,w):
        return quadratique_grad(self.data,self.y,w).sum(0)/len(self.data)

class PerceptronPlugin(Perceptron):
    def __init__(self,eps=1e-4,max_iter=5000,delta=1e-6,l=0):
        super(PerceptronPlugin,self).__init__(eps,max_iter,delta)
        self.l = l
    def _plugin_f(self,data,y,w):
        return ((1. - y*data.dot(w))**2) + np.linalg.norm(w**2)*self.l
    def _plugin_grad(self,data,y,w):
        return (2*(1-y*data.dot(w))*(-y*data.T)).T + 2.*self.l*w
    def f(self,w):
        return self._plugin_f(self.data,self.y,w).sum()
    def grad_f(self,w):
        # w = np.array(map(lambda x : max(0., 0.001 - x), w))
        return self._plugin_grad(self.data,self.y,w).sum(0)

def projection(data):
    return np.array([ [1,record[0],record[1],record[0]*record[1],record[0]**2,record[1]**2] for record in data ])

def phiGaussien(X,D,var):
    return np.array([ np.exp(-np.sum((record - D)**2,axis=1)/var) for record in X ])
'''
t   = 0
var = 0.5
trainX,trainY = gen_arti(data_type=t)
testX ,testY  = gen_arti(data_type=t)

trainX6D = projection(trainX)
testX6D = projection(testX)
perc   = PerceptronPlugin()
perc.l = 1.
perc.eps=1e-5
perc.fit(trainX6D,trainY)
plot(testX,testY, lambda x: perc.predict(np.array([projection(x)])))
print perc.score(testX6D,testY)
exit()

trainXGau = phiGaussien(trainX,trainX,var)
testXGau  = phiGaussien(testX,trainX,var)
# perc      = Perceptron()
# perc = PerceptronQuad(delta=1e-3)
perc = PerceptronPlugin()
# perc.eps=1e-10
perc.fit(trainXGau,trainY)
plot(testX,testY,lambda x: perc.predict(np.array([phiGaussien(x,trainX,var)])),50)
print perc.score(testXGau,testY)
exit()
''' 

for t in [5]:
    trainX,trainY = gen_arti(data_type=t)
    testX ,testY  = gen_arti(data_type=t)
    '''
    oldScore = 0
    for i in range(10):
        trainX6D = projection(trainX)
        testX6D = projection(testX)
        perc = Perceptron()
        perc.fit(trainX6D,trainY)
        print "hing loss score: ",perc.score(testX6D,testY)
        w = perc.x
        perc = PerceptronPlugin()
        perc.x = w
        perc._reset=False
        perc.eps = 1e-6
        perc.l   = 0.7
        perc.fit(trainX6D,trainY)
        score = perc.score(testX6D,testY)
        print "Plugin score: ", score
        if score > oldScore:
            oldScore = score
            oldPerc  = perc
    fname="./Plugin/polynomial[dataType=%d]"%(t)
    print fname
    plot(testX,testY,lambda x: oldPerc.predict(np.array([projection(x)])),50,fname)
    '''
    interv = np.arange(0.1,1.6,0.5)
    for var in interv:
        trainXGau = phiGaussien(trainX,trainX,var)
        testXGau  = phiGaussien(testX,trainX,var)
        perc = Perceptron()
        perc.fit(trainXGau,trainY)
        fname="./hingeLoss/phiGaussien[var=%d][dataType=%d]"%(var*10,t)
        plot(testX,testY,lambda x: perc.predict(np.array([phiGaussien(x,trainX,var)])),50,fname)
        print "hinge loss score: ",perc.score(testXGau,testY)
        w = perc.x
        perc = PerceptronPlugin()
        perc.x = w
        perc._reset=False
        # perc.eps = 1e-6
        perc.l   = 0.7
        perc.fit(trainXGau,trainY)
        fname="./Plugin/phiGaussien[var=%d][dataType=%d]"%(var*10,t)
        print fname
        plot(testX,testY,lambda x: perc.predict(np.array([phiGaussien(x,trainX,var)])),50,fname)
        print perc.score(testXGau,testY)


