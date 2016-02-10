# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:51:34 2015

@author: rbanderson
"""
from sklearn import cross_validation
import numpy as np
def random(df,nfolds=5,seed=10,groupby=None):
    df['Folds']='None'
    foldslist=np.array(df['Folds'])
    if groupby==None:
        n=len(df.index)
        folds=cross_validation.KFold(n,nfolds,shuffle=True,random_state=seed)
        i=1        
        for train,test in folds:
            foldslist[test]='Fold'+str(i)
            i=i+1
    
    else:
        unique_inds=np.unique(df[groupby])
        folds=cross_validation.KFold(len(unique_inds),nfolds,shuffle=True,random_state=seed)
        foldslist=np.array(df['Folds'])
        i=1        
        for train,test in folds:
            tmp=unique_inds[test]
            tmp_full_list=np.array(df[groupby])
            tmp_ind=np.in1d(tmp_full_list,tmp)
            foldslist[tmp_ind]='Fold'+str(i)
            i=i+1
    
    df['Folds']=foldslist
    return df