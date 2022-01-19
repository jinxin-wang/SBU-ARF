# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import pydot  #pour l'affichage graphique d'arbres
import matplotlib.pyplot as plt

from arftools import *

###############################
# Fonctions auxiliaires
###############################

eps = np.finfo(float).eps

def p_log_p(counts):
    """ fonction pour calculer \sum p_i log(p_i) """
    return np.sum(counts*np.log2(counts+eps))

def entropy(y):
    """ calcul de l'entropie d'un ensemble, attention c'est lent! """
    ylen = float(y.size)
    if ylen <= 1:
        return 0
    counts = np.array(Counter(y).values())/ylen
    return -p_log_p(counts)

###############################
# Classes
###############################

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
       """ A COMPLETER """
       return (self.predict(x) == y).mean()
    
class Split(object):
    """ Permet de coder un split pour une variable continue
        Contient :
            * le numero de la variable ou se fait le split
            * le seuil du split
            * le gain d'information du split
         predict(x) renvoie -1 si x_i<=seuil, +1 sinon
         best_gain(x,y) calcul le meilleur seuil pour la colonne x (1-dimension) et les labels y
         find_best_split(x,y) calcul le meilleur split pour les données x (n-dimensions) et les labels y
     """
    def __init__(self,idvar=None,threshold=None,gain=None):
        self.idvar=idvar
        self.threshold=threshold
        self.gain=gain

    def predict(self,x):
        if len(x.shape)==1:
            x=x.reshape((1,x.shape[0]))
        return np.array([-1 if x[i,self.idvar]<=self.threshold else 1 for i in range(x.shape[0])])

    @staticmethod
    def best_gain(x,y):
        ylen = float(y.size)
        idx_sorted = np.argsort(x)
        h=entropy(y)
        counts = Counter(y)
        dic_labs=dict(zip(counts.keys(),range(len(counts))))
        rcount = np.array(counts.values())
        lcount = np.zeros(len(counts))
        lc=0.
        rc=ylen
        xlast=x[idx_sorted[0]]
        split_val=x[idx_sorted[0]]
        hmin = h
        for i in range(y.size):
            if x[idx_sorted[i]]!=xlast:
                htmp = i /ylen*(-p_log_p(lcount/lc))+(ylen-i)/ylen*(-p_log_p(rcount/rc))
                if htmp<hmin:
                    hmin=htmp
                    split_val=(xlast+x[idx_sorted[i]])/2.
            rcount[dic_labs[y[idx_sorted[i]]]]-=1
            lcount[dic_labs[y[idx_sorted[i]]]]+=1
            lc+=1
            rc+=1
            xlast=x[idx_sorted[i]]
        return (h-hmin),split_val

    @staticmethod
    def find_best_split(x,y):
        if len(x.shape)==1:
            x = x.reshape((1,x.shape[0]))
        hlist = [[Split.best_gain(x[:,i],y),i] for i in range(x.shape[1])]
        (h,threshold),idx= max(hlist)
        return Split(idx,threshold,h)

    def __str__(self):
        return "var %s, thresh %f (gain %f)" %(self.idvar,self.threshold, self.gain)
    def __repr__(self):
        return self.__str__()

class Node(Classifier):
    """ Noeud d'un arbre
        split contient le split du noeud
        parent,left, right les noeuds parents, gauche et droit du noeud (parent = None si racine, left=right=None si feuille)
        leaf indique si le noeud est une feuille
        depth la profondeur du noeud
        label le label majoritaire du noeud
        info est un dictionnaire pour stocker des informations supplemantaires
    """
    def __init__(self,split=None,parent=None,left=None,right=None,leaf=True,depth=-1,label=None,**kwargs):
        self.split=split
        self.parent=None
        self.left=None
        self.right=None
        self.info=dict(kwargs)
        self.leaf=leaf
        self.label=label
        self.depth=depth

    def predict(self,x):
        if len(x.shape)==1:
            x=x.reshape((1,x.shape[0]))
        if self.leaf:
            return np.array([self.label]*x.shape[0])
        return np.array([self.left.predict(x[i,:])[0] if res<0 else self.right.predict(x[i,:])[0] for i,res in enumerate(self.split.predict(x))])

    def fit(self,x,y):
        counts=Counter(y)
        self.split=Split.find_best_split(x,y)
        self.label = counts.most_common()[0][0]

    def __str__(self):
        if self.leaf:
            return "Leaf : %s" % (self.label,)
        return "Node : %s (%s)" % (self.split,self.info)
    def __repr__(self):
        return self.__str__()

class DecisionTree(Classifier):
    """ Arbre de decision
    max_depth indique la profondeur max de l'arbre
    min_samples_split le nombre minimal d'exemples pour continuer l'apprentissage
    to_dot permet de convertir en dot l'arbre (affichage graphique)
    to_pdf d'enregistrer l'arbre dans un fichier pdf
    """

    def __init__(self,max_depth=5,min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split

    def fit(self,x,y):
        """ apprentissage de l'arbre de maniere iterative
        on apprend un noeud, puis on cree les deux enfants de ce noeud, que l'on ajoute a la pile des noeuds
        a traiter par la suite (nodes_to_treat), ainsi que les index des exemples associes (dic_idx)
        """

        self.root=Node(depth=0)
        nodes_to_treat = [self.root]
        dic_idx=dict({self.root : range(len(y))})
        while len(nodes_to_treat)>0:
            # recuperation du noeud courant
            curnode = nodes_to_treat.pop()
            #recuperation de la liste des indices des exemples associes, x[idx_train,:] contient l'ensemble des
            #exemples a traiter
            idx_train = dic_idx.pop(curnode)
            # infos complementaires sur le nombre d'exemples en apprentissage par label
            for lab,clab in Counter(y[idx_train]).items():
                curnode.info[lab]=clab
            # A COMPLETER #
            #trouve le meilleur split pour ce noeud
            curnode.fit(x[idx_train],y[idx_train])
            
            # recupere les predictions pour partager entre fils droit et gauche les exemples
            pred =  curnode.split.predict(x[idx_train,:])
            l_idx = [ idx_train[i] for i in range(len(idx_train)) if pred[i]<0 ]
            r_idx = list(set(idx_train).difference(l_idx))

            #Condition d'arrets
            if entropy(y[idx_train])==0 or curnode.depth >= self.max_depth or \
                    len(l_idx) < self.min_samples_split or len(r_idx) < self.min_samples_split:
                curnode.leaf=True
                continue
            #Creation des deux enfants
            curnode.left = Node(parent=curnode,depth=curnode.depth+1)
            curnode.right = Node(parent=curnode,depth=curnode.depth+1)
            curnode.leaf=False
            #On enregistre les indices correspondant aux deux noeuds
            dic_idx[curnode.left]=l_idx
            dic_idx[curnode.right]=r_idx
            #On ajoute les deux enfants a la liste des noeuds a traiter
            nodes_to_treat = [curnode.left,curnode.right]+nodes_to_treat

    def predict(self,x):
        # A COMPLETER
        return self.root.predict(x)
    
    def __str__(self):
        s=""
        nodes=[self.root]
        while len(nodes)>0:
            curnode=nodes.pop()
            if not curnode.leaf:
                s+= "\t"*curnode.depth + "var %d :  <=|> %f \n"  %(curnode.split.idvar,curnode.split.threshold)
                nodes+=[curnode.left,curnode.right]
            else:
                s+= "\t"*curnode.depth + "class : %s\n" %(curnode.label,)
        return s
        
    def __repr__(self):
        return self.__str__()

    def to_dot(self,dic_var=None):
        s="digraph Tree {"
        cpt=0
        nodes = [(self.root,cpt)]
        while len(nodes)>0:
            curnode,idx = nodes.pop()
            labinfo = ",".join(["%s: %s" % (lab,slab) for lab,slab in curnode.info.items()])
            if not curnode.leaf:
                s+="%d [label=\"%s <= %f\n IG=%f\n " %(idx,curnode.split.idvar \
                    if not dic_var else dic_var[curnode.split.idvar],curnode.split.threshold,curnode.split.gain)
                s+= " %s \n \",shape=\"box\" ];\n"  % (labinfo,)
                lidx = cpt +1
                ridx = cpt +2
                s+= "%d -> %d; %d -> %d;\n" % (idx,lidx,idx,ridx)
                cpt+=2
                nodes += [(curnode.left,lidx),(curnode.right,ridx)]
            else:
                s+= "%d [label=\"label=%s\n %s \"];\n" %(idx,curnode.label,labinfo)
        return s+"}"
    def to_pdf(self,filename,dic_var=None):
        pydot.graph_from_dot_data(self.to_dot(dic_var)).write_pdf(filename)

def plot_scores(arange,repeat,xTrain,yTrain,xTest,yTest):
    scores = []
    for depth in arange:
        s = []
        for i in range(repeat):
            dt = DecisionTree(max_depth=depth)
            dt.fit(xTrain,yTrain)
            s.append(dt.score(xTest,yTest))
        scores.append(np.mean(s))

    fig = plt.plot(range(2,16),scores,'b-')
    plt.xlabel("Depth of Decision Tree")
    plt.ylabel("Score")
    plt.show()
    plt.close(fig)

'''
xTrain,yTrain = gen_arti(data_type=4)
xTest, yTest  = gen_arti(data_type=4)

# plot_scores(range(2,16),10,xTrain,yTrain,xTest,yTest)

dt = DecisionTree(15)
dt.fit(xTrain,yTrain)
plot_4class(xTest,yTest,dt.predict,step=100)
print dt.score(xTest,yTest)

exit()
plot_frontiere(xTest,dt.predict,step=1000)
plot_data(xTest,yTest)
plt.show()
print dt.__repr__()
'''
