# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:58:07 2015

@author: rbanderson
This is a simple function to read in CSV data.
If setindex is specified, then it uses the columnd of the CSV with the 
specified name as the row index of the data frame
"""
import pandas as pd
def CSV(filename,sep=',',setindex=None):
    print('Reading '+filename)
    df = pd.read_csv(filename, sep=sep)
    wvlindex=[]
    cols_wvl=[]
    nonwvlindex=[]
    for i,x in enumerate(df.columns):
        try:
            x=round(float(x),5)
            cols_wvl.append(('wvl',x))
            wvlindex.extend([i])
        except:
            nonwvlindex.extend([i])
    
    df_spectra=df[wvlindex]
    df_data=df[nonwvlindex]
    df_spectra.columns=pd.MultiIndex.from_tuples(cols_wvl)
    for i,x in enumerate(df_data.columns):
        df_spectra[x]=df_data[x]
    df=df_spectra

    if setindex:
        df=df.set_index([setindex])


    return df