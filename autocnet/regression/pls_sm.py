# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:15:46 2016

@author: rbanderson
"""
import numpy as np
import autocnet.spectral.within_range as within_range
from sklearn.cross_decomposition.pls_ import PLSRegression
from autocnet.spectral.meancenter import meancenter


    
    

def fit(datasets,ranges,ncs,ycol):
    submodels=[]    
    mean_vects=[]
    for n,i in enumerate(ranges):
        data_tmp=within_range(datasets[i],n,ycol)
        x=data_tmp['wvl']
        y=data_tmp['meta'][ycol]
        x_centered,x_mean_vect=meancenter(x) #mean center training data
        pls=PLSRegression(n_components=ncs[i],scale=False)
        pls.fit(x,y)
        submodels.append(pls)
        mean_vects.append(x_mean_vect)
    
        E=x_centered-np.dot(pls.x_scores_,pls.x_loadings_.transpose())
        Q_res=np.dot(E,E.transpose()).diagonal()
        T=pls.x_scores_
        #There's probably a more efficient way to calculate T2...
        for k in range(len(x_centered[:,0])):
            T2[k]=np.dot(T[k],np.dot(np.linalg.inv(np.dot(T.transpose(),T)),T[k]))
       plot.figure()
       plot.scatter(T2,Q_res.'o','r')
        print(foo)
    return submodels,mean_vects
            
     
def blend(testset,models,ranges,refmodel,toblend,mean_vects):
    predicts=[]
    for pls_temp in models:
        print(foo)
        

#    print('do stuff here')
#
#def transform(self,x):
#    print('do stuff here')

