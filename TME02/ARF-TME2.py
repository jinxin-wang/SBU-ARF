# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from collections import Counter

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    # center : entre des gaussiennes
    # sigma : ecart type des gaussiennes
    # nbex : nombre d'exemples
    # ex_type : vrai pour gaussiennes, faux pour echiquier
    # epsilon : bruit
    
    if data_type==0:
        #melange de 2 gaussiennes
        xpos=np.random.multivariate_normal([centerx,centery],np.diag([sigma,sigma]),nbex/2)
        xneg=np.random.multivariate_normal([-centerx,-centery],np.diag([sigma,sigma]),nbex/2)
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
        
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centery],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([-centerx,-centery],np.diag([sigma,sigma]),nbex/4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centery],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([centerx,-centery],np.diag([sigma,sigma]),nbex/4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
        
    if data_type==2:
        # melange de 16 gaussiennes
        data = None
        y    = None
        for wx,wy in [[1,1],[1,4],[4,1],[4,4]]:
            xpos=np.vstack((np.random.multivariate_normal([ centerx*wx, centery*wy],np.diag([sigma,sigma]),nbex/4),
                            np.random.multivariate_normal([-centerx*wx,-centery*wy],np.diag([sigma,sigma]),nbex/4)))
            xneg=np.vstack((np.random.multivariate_normal([-centerx*wx, centery*wy],np.diag([sigma,sigma]),nbex/4),
                            np.random.multivariate_normal([ centerx*wx,-centery*wy],np.diag([sigma,sigma]),nbex/4)))
            if data is None or y is None:
                data=np.vstack((xpos,xneg))
                y=np.hstack((np.ones(nbex/2),
                            -np.ones(nbex/2)))
            else:
                data=np.vstack((data,xpos,xneg))
                if wx == wy:
                    y=np.hstack((y,np.ones(nbex/2),
                                  -np.ones(nbex/2)))
                else:
                    y=np.hstack((y,-np.ones(nbex/2),
                                    np.ones(nbex/2)))
                    
    if data_type==3:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
        
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
    
# plot_data(data,y)

#Frontiere de decision
def plot_frontiere(x,y,f,pltname=None,step=20): # script qui engendre une grille sur l'espace des exemples, calcule pour chaque point le label
                                 # et trace la frontiere
    mmax=x.max(0)
    mmin=x.min(0)
    fig = plt.figure()
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    # calcul de la prediction pour chaque point de la grille
    res=np.array([f(grid[i,:]) for i in range(x1grid.size)])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=('gray','blue'),levels=[-1,0,1])
    plot_data(x,y)
    if pltname == None:
        plt.show()
    else:
        fig.savefig(pltname)
        fig.clear()
        
class Classifier(object):
    """ Classe generique d'un classifieur
        Dispose de 3 méthodes :
            fit pour apprendre
            predict pour predire
            score pour evaluer la precision
    """
    def fit(self,X,Y):
        raise NotImplementedError("fit non implemente")
    def predict(self,X):
        raise NotImplementedError("predict non implemente")
    def score(self,X,Y):
        if len(np.shape(Y)) == 0:
            return (self.predict(x) == y).mean()
            
        if len(X) <> len(Y):
            raise ValueError("Taille de Test Set est unmatched.")
            
        return np.array([self.predict(X[i]) == Y[i] for i in range(len(X))]).mean()
    
class Knn(Classifier):
    def __init__(self, k):
        self.k = k
        
    def fit(self,trX,trY):
        self.trX = trX
        self.trY = trY
        
    def predict(self,X):
        # return np.array([ 1 if self.trY[np.array([ np.linalg.norm(X - x) for x in self.trX ]).argsort()[1:self.k+1]].sum() > 0 else -1 for X in tstX ])
        D = np.array([ np.linalg.norm(X - x) for x in self.trX ])
        sortInd = D.argsort()[:self.k]
        if self.trY[sortInd].sum() > 0:
            return 1
        else:
            return -1

class Parzen(Classifier):
    def __init__(self, K, h):
        self.K = K # Kernel function
        self.h = h # taille de fenetre sur 2
        
    def fit(self,trX,trY):
        self.trX = trX
        self.trY = trY
        
    def predict(self,X):
        D = np.array([ self.K(X,x,self.h) for x in self.trX ])
        if self.trY[D==1].sum() > 0:
            return 1
        else:
            return -1
        
def hypercube(x0,x,h):
    return np.prod(np.absolute(x0 - x) < h)

def sphere(x0,x,h):
    return np.linalg.norm(x0 - x) < h

def gauss(x0,x,h):
    u = np.linalg.norm(x0 - x)
    return np.sqrt(np.pi*2) * np.exp(-(u**2)/2) < h

def laplace(x0,x,h):
    u = np.linalg.norm(x0 - x)
    return np.exp(-np.absolute(u))/2 < h

def epanechikov(x0,x,h):
    u = np.linalg.norm(x0 - x)
    return 3*max((0,1-u**2))/4 < h

def uniform(x0,x,h):
    u = np.linalg.norm(x0 - x)
    a = -1
    b = 1
    if u > b:
        khi = 1
    elif u < a:
        khi = 0
    else:
        khi = (u - a)/(b-a)
    return khi/2 < h

'''
t = 0

trainX,trainY = gen_arti(data_type=t)
testX ,testY  = gen_arti(data_type=t)

# for k in [2**i for i in range(10,14)] :
k = len(trainY)
knn = Knn(k)
knn.fit(trainX,trainY)
# print "function: data_type[%d] Knn[%d]\t Score: %f\n"%(t,k,knn.score(testX,testY))
# plot_frontiere(testX,testY,knn.predict)
fname = "knn[k=%d][data_type=%d][score=%f].png"%(k,t,knn.score(testX,testY))
print fname
plot_frontiere(testX,testY,knn.predict,fname)

for K in [hypercube,sphere,gauss,laplace,epanechikov,uniform]:
    for h in (0.7,0.5,0.3,0.1,0.05):
        parzen = Parzen(K,h)
        parzen.fit(trainX,trainY)
        # print "function: %s[%2f]\t Score: %f\n"%(K.__name__,h,parzen.score(testX,testY))
        fname = "parzen[K=%s][h=%2f][score=%f][data_type=%d].png"%(K.__name__,h,parzen.score(testX,testY),2)
        print fname
        plot_frontiere(testX,testY,parzen.predict,fname)
        # plot_frontiere(testX,testY,parzen.predict)
'''
arange = [0.001*(2**i) for i in range(10) ] # np.arange(0.001,0.52,0.01)
for t in [0,2,3]:
    trainX,trainY = gen_arti(data_type=t)
    testX ,testY  = gen_arti(data_type=t)

    for K in [gauss]: # [hypercube,sphere,gauss]: #,laplace,epanechikov,uniform]:
        scores = []
        fname = "parzen[K=%s][data_type=%d].png"%(K.__name__,t)
        for h in arange:
            parzen = Parzen(K,h)
            parzen.fit(trainX,trainY)
            scores.append(parzen.score(testX,testY))
        fig = plt.figure()
        plt.plot(arange,scores)
        plt.ylabel("Scores")
        plt.xlabel("Taille de Fenetre")
        plt.savefig(fname)
        plt.close(fig)
        print fname
        
