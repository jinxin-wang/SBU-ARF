# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
import matplotlib.pyplot as plt



def to_array(x):
    """ convert an vector to array if needed """
    if len(x.shape)==1:
        x=x.reshape(1,x.shape[0])
    return x

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    #center : entre des gaussiennes
    #sigma : ecart type des gaussiennes
    #nbex : nombre d'exemples
    # ex_type : vrai pour gaussiennes, faux pour echiquier
    #epsilon : bruit

    if data_type==0:
        #melange de 2 gaussiennes
        xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex/2)
        xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex/2)
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
    if data_type==4:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.zeros(nbex/4),2*np.ones(nbex/4)))
        y=np.hstack((y, 3*np.ones(nbex/4)))
        y=np.hstack((y, np.ones(nbex/4)))
        y=y+1
    if data_type==2:
        # melange de 16 gaussiennes
        data = None
        y    = None
        for wx,wy in [[1,1],[1,4],[4,1],[4,4]]:
            xpos=np.vstack((np.random.multivariate_normal([centerx*wx,centery*wy],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([-centerx*wx,-centery*wy],np.diag([sigma,sigma]),nbex/4)))
            xneg=np.vstack((np.random.multivariate_normal([-centerx*wx,centery*wy],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([centerx*wx,-centery*wy],np.diag([sigma,sigma]),nbex/4)))
            if data is None or y is None:
                data=np.vstack((xpos,xneg))
                y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
            else:
                data=np.vstack((data,xpos,xneg))
                if wx == wy:
                    y=np.hstack((y,np.ones(nbex/2),-np.ones(nbex/2)))
                else:
                    y=np.hstack((y,-np.ones(nbex/2),np.ones(nbex/2)))
    if data_type==3:
        # echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    if data_type==5:
        # echiquier
        data0=np.random.multivariate_normal([ 1.5, 1.5],[[0.5,-0.5],[1,0.5]],nbex/2)
        data1=np.random.multivariate_normal([-1.5,-1.5],[[0.5,-0.5],[1,0.5]],nbex/2)
        data =np.vstack((data0,data1))
        y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,len(data))
    data[:,1]+=np.random.normal(0,epsilon,len(data))
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y

    #affichage en 2D des donnees
def plot_data(x,labels):
        plt.scatter(x[labels<0,0],x[labels<0,1],c='red',marker='x')
        plt.scatter(x[labels>0,0],x[labels>0,1],c='green',marker='+')
        
def plot_data_4class(x,labels):
        plt.scatter(x[labels==1,0],x[labels==1,1],c='green',marker='+')
        plt.scatter(x[labels==2,0],x[labels==2,1],c='yellow',marker='o')
        plt.scatter(x[labels==3,0],x[labels==3,1],c='pink',marker='*')
        plt.scatter(x[labels==4,0],x[labels==4,1],c='red',marker='x')

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,data=None,step=20):
    if data is None:
        xmax=np.max(data[:,0])
        xmin=np.min(data[:,0])
        ymax=np.max(data[:,1])
        ymin=np.min(data[:,1])
    x=np.arange(xmin,xmax,(xmax-xmin)*1./step)
    y=np.arange(ymin,ymax,(ymax-ymin)*1./step)
    xx,yy=np.meshgrid(x,y)
    grid=np.c_[xx.ravel(),yy.ravel()]
    return grid,xx,yy

#Frontiere de decision
def plot_frontiere(x,f,step=20):
        grid,xvec,yvec=make_grid(data=x,step=step)
        res=f(grid)
        res=res.reshape(xvec.shape)
        plt.contourf(xvec,yvec,res,colors=('gray','blue'),levels=[-1,0,1])

def plot_frontiere_4class(x,f,step=20):
        grid,xvec,yvec=make_grid(data=x,step=step)
        res=f(grid)
        res=res.reshape(xvec.shape)
        plt.contourf(xvec,yvec,res,colors=('gray','blue','orange','cyan'),levels=[-1,1,2,3,4])

def plot(x,labels,f,step=20,fname=None):
    fig = plt.figure()
    plot_frontiere(x,f,step)
    plot_data(x,labels)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close(fig)

def plot_4class(x,labels,f,step=20,fname=None):
    fig = plt.figure()
    plot_frontiere_4class(x,f,step)
    plot_data_4class(x,labels)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close(fig)

def traceVraisemblance(vList,fname=None):
    fig = plt.figure()
    plt.plot()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close(fig)

def traceEspaceDesCouts(X,y,allw,wstar):
    # tracer de l'espace des couts
    ngrid = 20
    w1range = np.linspace(-14, 14, ngrid)
    w2range = np.linspace(-15, 15, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)
    cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-y)**2).sum()) for w1i in w1range] for w2j in w2range])
    fig = plt.figure()
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1],c='r')
    plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
    plt.show()
    plt.close(fig)

##################################################################

class Classifier(object):
    """ Classe generique d'un classifieur
        Dispose de 3 méthodes :
            fit pour apprendre
            predict pour predire
            score pour evaluer la precision
    """

    def fit(self,x,y):
        raise NotImplementedError("fit non  implemente")
    def predict(self,x):
        raise NotImplementedError("predict non implemente")
    def score(self,x,y):
        return (self.predict(x)==y).mean()

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
        self.eps     =eps
        self.optim_f =optim_f
        self.max_iter=max_iter
        self.delta   =delta
    def reset(self):
        self.i=0
        self.x = self.optim_f.x_random()
        self.log_x=np.array(self.x)
        self.log_f=np.array(self.optim_f.f(self.x))
        self.log_grad=np.array(self.optim_f.grad_f(self.x))
    def optimize(self,_reset=True):
        if _reset:
            self.reset()
        while not self.stop():
            self.x    = self.x - self.get_eps()*self.optim_f.grad_f(self.x)
            self.log_x=np.vstack((self.log_x,self.x))
            self.log_f=np.vstack((self.log_f,self.optim_f.f(self.x)))
            self.log_grad=np.vstack((self.log_grad,self.optim_f.grad_f(self.x)))
            self.i+=1
    def stop(self):
        return (self.i>2) and (self.max_iter and (self.i>self.max_iter) or (self.delta and np.abs(self.log_f[-1]-self.log_f[-2]))<self.delta)
    def get_eps(self):
        return self.eps
