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

class OptimFunc(object):
    def __init__(self,f=None,grad_f=None,dim=2):
        self._f=f
        self._grad_f=grad_f
        self.dim=dim
    def x_random(self,low=-5,high=5):
        return random.random(self.dim)*(high-low)+low
    def f(self,x):
        return self._f(to_array(x))
    def grad_f(self,x):
       return self._grad_f(to_array(x))

class GradientDescent(object):
    def __init__(self,optim_f,eps=1e-4,max_iter=5000,delta=1e-6):
        self.eps=eps
        self.optim_f=optim_f
        self.max_iter=max_iter
        self.delta=delta
    def reset(self):
        self.i=0
        self.x = self.optim_f.x_random()
        self.log_x=np.array(self.x)
        self.log_f=np.array(self.optim_f.f(self.x))
        self.log_grad=np.array(self.optim_f.grad_f(self.x))
    def optimize(self,reset=True):
        if reset:
            self.reset()
        while not self.stop():
            self.x = self.x - self.get_eps()*self.optim_f.grad_f(self.x)
            self.log_x=np.vstack((self.log_x,self.x))
            self.log_f=np.vstack((self.log_f,self.optim_f.f(self.x)))
            self.log_grad=np.vstack((self.log_grad,self.optim_f.grad_f(self.x)))
            self.i+=1
    def stop(self):
        return (self.i>2) and (self.max_iter and (self.i>self.max_iter) or (self.delta and np.abs(self.log_f[-1]-self.log_f[-2]))<self.delta)
    def get_eps(self):
        return self.eps

# <markdowncell>

# Un exemple d'utilisation est le suivant, pour la fonction f(x)=x.cos(x) :

# <codecell>


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
        self.optimize()
    def f(self,w):
        return hinge_f(self.data,self.y,w).sum()
    def grad_f(self,w):
        return hinge_grad(self.data,self.y,w).sum(0)
    def predict(self,testX):
        X = to_array(testX)
        return np.sign(X.dot(self.x))

def projection(data):
    return np.array([ [1,record[0],record[1],record[0]*record[1],record[0]**2,record[1]**2] for record in data ])

def phiGaussien(x,data,var):
    return np.array([ np.exp(-(np.linalg.norm(record - data,axis=1)**2)/var) for record in x ])

class SVM(Classifier,OptimFunc,GradientDescent):
    def __init__(self,eps=1e-4,max_iter=5000,delta=1e-6):
        GradientDescent.__init__(self,self,eps,max_iter,delta)
    def fit(self,data,y):
        self.dim  = len(data[0])
        self.data = data
        self.y    = y
        self.optimize()
    def f(self,w):
        return hinge_f(self.data,self.y,w).sum()
    def grad_f(self,w):
        return hinge_grad(self.data,self.y,w).sum(0)
    def predict(self,testX):
        X = to_array(testX)
        return np.sign(X.dot(self.x))

