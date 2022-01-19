# -*- coding: utf-8 -*-

import numpy as np
from arftools import *


class Kmeans(object):
    def __init__(self,K):
        self.K = K
        self.RESET = True
        self.STOP  = False
    def fit(self, X, Y):
        self.data = X
        self.label= Y
        self.dim  = len(X[0])
        self.scale= np.max(X,axis=0)
        self.optimize()

    def init_prototype(self):
        self.prototype = np.random.randn(self.K,self.dim)*self.scale
        
    def optimize(self):
        # Initialisation des k Centres
        if self.RESET:
            self.init_prototype()

        self.iCount = 0
        while not self.STOP:
            self.iCount = self.iCount + 1
            # Affectation des points
            C = self.predict(self.data)
            # Estimation des centres
            self.update_prototype(C)
            print self.prototype
            if self.iCount > 10:
                self.STOP = True

    def update_prototype(self,C):
        # self.prototype = np.array([ self.class_cost(i,C==i) for i,mu in enumerate(self.prototype)])
        self.prototype = [ np.mean(self.data[C==i],axis=0).tolist() for i in range(len(self.prototype))]
        
    def score(self):
        pass

    def predict(self,X):
        return np.argmin([ [ np.linalg.norm(x-mu)**2 for mu in self.prototype ] for x in X ], axis=1)

    def class_cost(self,class_indice, data_indice):
        mu = self.prototype[class_indice]
        return np.sum((self.data[data_indice]-mu)**2)/len(data_indice)
        # return np.sum([ [ np.sum((x-mu)**2) ] for x in self.data[data_indice]], axis=0)/sum(data_indice)
    
    def cost(self):
        return np.sum([ [ np.linalg.norm(x-mu) for mu in self.prototype ] for x in self.data ])

'''            
t = 0
X,Y = gen_arti(data_type=t)

cluster = Kmeans(2)
cluster.fit(X,Y)
plot(X,Y,cluster.predict)
'''
    
class ImgKmeans(Kmeans):
    def __init__(self,K,picName):
        super(ImgKmeans,self).__init__(K)
        self.picName = picName
    def imshow(self,picM,fname=None):
        fig = plt.figure()
        plt.imshow(picM)
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)
        plt.close(fig)
        
    def imread(self):
        return plt.imread(self.picName)



picName = "grass.png"    
