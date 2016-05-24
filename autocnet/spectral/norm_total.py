# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:20:38 2016

@author: rbanderson


"""
import numpy as np
import pandas as pd

def norm_total(df):
    df=df.div(df.sum(axis=1),axis=0)
    return df
    
def norm_spect(df,ranges):
    df_spect=df['wvl']
    df_meta=df['meta']
    wvls=df_spect.columns.values
    df_sub_norm=[]
    allind=[]    
    for i in ranges:
        #Find the indices for the range
        ind=(np.array(wvls,dtype='float')>=i[0])&(np.array(wvls,dtype='float')<=i[1])
        #find the columns for the range
        cols=wvls[ind]
        #keep track of the indices used for all ranges
        allind.append(ind)
        #add the subset of the full df to a list of dfs to normalize
        df_sub_norm.append(norm_total(df_spect[cols]))
    
    #collapse the list of indices used to a single array
    allind=np.sum(allind,axis=0)
    #identify wavelengths that were not used by where the allind array is less than 1
    wvls_excluded=wvls[np.where(allind<1)]
    #create a separate data frame containing the un-normalized columns
    df_excluded=df_spect[wvls_excluded]
    
    #combine the normalized data frames into one
    df_norm=pd.concat(df_sub_norm,axis=1)
    
    #make the columns into multiindex
    df_excluded.columns=[['masked']*len(df_excluded.columns),df_excluded.columns]    
    df_norm.columns=[['wvl']*len(df_norm.columns),df_norm.columns.values] 
    df_meta.columns=[['meta']*len(df_meta.columns),df_meta.columns.values]
    
    #combine the normalized data frames, the excluded columns, and the metadata into a single data frame
    df_new=pd.concat([df_meta,df_norm,df_excluded],axis=1)
    
    return df_new