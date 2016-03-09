# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:11:11 2016

@author: rbanderson
"""
import numpy as np
from autocnet.spectral.baseline_code.als import ALS
from autocnet.spectral.baseline_code.dietrich import Dietrich
from autocnet.spectral.baseline_code.polyfit import PolyFit
from autocnet.spectral.baseline_code.airpls import AirPLS
from autocnet.spectral.baseline_code.fabc import FABC
from autocnet.spectral.baseline_code.kajfosz_kwiatek import KajfoszKwiatek as KK
from autocnet.spectral.baseline_code.mario import Mario
from autocnet.spectral.baseline_code.median import MedianFilter
from autocnet.spectral.baseline_code.rubberband import Rubberband
#from autocnet.spectral.baseline_code.wavelet import Wavelet

def remove_baseline(df,method='als',segment=True,params=None):
    wvls=np.array(df['wvl'].columns.values,dtype='float')
    spectra=np.array(df['wvl'],dtype='float')
    
   
    #set baseline removal object (br) to the specified method
    if method is 'als':
        br=ALS()
    if method is 'dietrich':
        br=Dietrich()
    if method is 'polyfit':
        br=PolyFit()
    if method is 'airpls':
        br=AirPLS()
    if method is 'fabc':
        br=FABC()
    if method is 'kk':
        br=KK()
    if method is 'mario':
        br=Mario()
    if method is 'median':
        br=MedianFilter()
    if method is 'rubberband':
        br=Rubberband()
    #if method is 'wavelet':
     #   br=Wavelet()
        
        
    #if parameters are provided, use them to set the parameters of br
    if params is not None:
        for i in br.__dict__.keys():
            try:
                br[i]=params[i]
            except:
                print('Required keys are:')
                print(br.__dict__.keys())
    br.fit(wvls,spectra,segment=segment)
    df_baseline=df.copy()
    df_baseline['wvl']=br.baseline
    df['wvl']=br.fit_transform(wvls,spectra)
    return df, df_baseline    
            
    