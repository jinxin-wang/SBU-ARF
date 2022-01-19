# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # TME 3 - Descente de gradient
# 
# Il s'agit dans ce TME de coder la descente de gradient pour une fonction générique.
# Nous utiliserons pour cela une classe abstraite OptimFunc, qui contiendra la définition de la fonction à calculer (f), le gradient de cette fonction (grad_f) et la fonction x_random pour générer aléatoirement un point initial.

# <codecell>

import numpy as np
from numpy import random
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fonction pour transformer un vecteur en array si ce n'est pas un array
#, a mettre en première ligne de l'appel de fonctions, comme dans le squelette OptimFunc
def to_array(x):
    if len(x.shape)==1:
        x=x.reshape(1,x.shape[0])
    return x

#Classe abstraite d'une fonction à optimiser. self.dim contient la dimension attendue de l'entrée (pour la génération aléatoire d'un point).

class OptimFunc(object):
    dim = 2    
    def f(self,x): # fonction a calculer
        x=to_array(x)
        pass
    
    def grad_f(self,x): # gradient de la fonction
        x=to_array(x)
        pass
    
    def x_random(self,low=0,high=1):
        return random.random(self.dim)*(high-low)+low

# <markdowncell>

# Voici le squelette de la classe pour l'optimisation par descente de gradient. Les paramètres sont les suivants:
# 
# * eps : epsilon, le pas du gradient; la fonction get_eps() permet de faire un pas adaptatif
# 
# * optiim_f : objet de type OptimFunc, la fonction à optimiser
# 
# * max_iter : le nombre d'itération maximum, None s'il ne faut pas le prendre en compte
# 
# * delta : la différence minimale à avoir entre deux pas pour continuer à itérer
# 
# En plus de ces paramètres, les variables suivantes sont utilisées :
#  
# * i : le numéro de l'itération courante
# 
# * x : le point optimal courant
# 
# * log_x, log_f, log_grad : l'historique des points parcourus, de la valeur de la fonction en ces points, de la valeur du gradient en ces points.
# 
# Compléter ce qui manque. Utiliser en particuliler np.vstack((log_x,x)) pour empiler au fur et à mesure les résultats dans l'historique.

# <codecell>

class GradientDescent(object):
    def __init__(self,optim_f,eps=1e-4,max_iter=5000,delta=1e-6):
        self.eps     = eps
        self.optim_f = optim_f
        self.max_iter= max_iter
        self.delta   = delta
        self.reset()

    def reset(self):
        self.i=0
        self.x = self.optim_f.x_random()
        self.log_x=np.array(self.x)
        self.log_f=np.array(self.optim_f.f(self.x))
        self.log_grad=np.array(self.optim_f.grad_f(self.x))

    def optimize(self,reset=True):
        if reset:
            self.reset()
            
        while (self.stop() <> True):
            self.i = self.i + 1
            # print self.x
            self.x = self.x - self.get_eps()*self.optim_f.grad_f(self.x)
            # print self.x
            self.log_x   = np.vstack((self.log_x,self.x))
            # print self.log_f.shape,self.optim_f.f(self.x).shape
            self.log_f   = np.vstack((self.log_f,self.optim_f.f(self.x)))
            self.log_grad=np.vstack((self.log_grad,self.optim_f.grad_f(self.x)))
            
    def stop(self):
        return (self.i>2) and ((self.max_iter and (self.i>self.max_iter))\
                               or ((self.delta and np.abs(self.log_f[-1]-self.log_f[-2]))<self.delta))

    def get_eps(self):
        return self.eps

# <markdowncell>

# # Mise en pratique
# Tester votre implémentation sur 3 fonctions :
# + en 1d sur la fonction f(x)=x.cos(x)
class Xcosx(OptimFunc):
    def __init__(self):
        self.dim = 1
        
    def f(self,x):
        x=to_array(x)
        return x*np.cos(x)

    def grad_f(self,x): # gradient de la fonction
        x=to_array(x)
        return np.cos(x) - x*np.sin(x)

'''
tObj1 = Xcosx()
gd = GradientDescent(tObj1)
print "gd.optimize(): "
gd.optimize()
'''
# + en 1d  sur la fonction -log(x)+x^2
class Gx(OptimFunc):
    def __init__(self):
        self.dim = 1

    def f(self,x):
        x = to_array(x)
        return -np.log(x)+x**2

    def grad_f(self,x):
        x = to_array(x)
        return -1./x + 2*x
'''
tObj2 = Gx()
gd = GradientDescent(tObj2)
print "gd.optimize(): "
gd.optimize()
'''

# + en 2d sur la fonction Rosenbrock (ou banana), définie par : f(x_1,x_2)= 100*(x_2-x_1^2)^2 + (1-x_1)^2 .
class Rosenbrock(OptimFunc):
    def __init__(self):
        self.dim = 2

    def f(self,x):
        x   = to_array(x)
        x_1 = x[:,0]
        x_2 = x[:,1]
        return 100*(x_2 - x_1**2)**2 + (1 - x_1)**2

    def grad_f(self,x):
        x = to_array(x)
        x_1 = x[:,0]
        x_2 = x[:,1]
        return np.array([200*(x_2-x_1**2)*(-2*x_1)-2*(1-x_1),200*(x_2-x_1**2)]).T

# Tracer les courbes d'évolution de f(x) en fonction du nombre d'itérations. Tracer (en 3d ou en 2d) la surface de la fonction, et la trajectoire de l'optimisation. Vous pouvez utilisez le code suivant pour la visualisation 3d.

# <codecell>

#Quadrillage pour evaluer la fonction f rosenbrock.
m=5
rosen = Rosenbrock()
x=np.arange(-m,m,2*m/20.)
y=np.arange(-m,m,2*m/20.)
xx,yy=np.meshgrid(x,y)
grid=np.c_[xx.ravel(),yy.ravel()]
z=rosen.f(grid).reshape(xx.shape)

grd_rosen = GradientDescent(rosen)
grd_rosen.optimize()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)

ax.plot(grd_rosen.log_x[:,0],grd_rosen.log_x[:,1],grd_rosen.log_f.ravel(),color='black')
    
plt.show()

# <markdowncell>

# # Régression logistique
# Implémenter la résolution pour la régression logistique. Tester sur les données artificielles et sur les données réèlles MNIST (reconnaissance des chiffres).

# <codecell>


