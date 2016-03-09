# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:53:23 2015

@author: rbanderson
"""
from autocnet.spectral.interp import interp_spect
from autocnet.spectral.mask import mask
from autocnet.utils.folds import random
from autocnet.spectral.norm_total import norm_spect
from autocnet.spectral.remove_baseline import remove_baseline

class spectral_data(object):
    def __init__(self,df):
        self.df=df
    
    def interp(self,*args,**kwargs):
        return spectral_data(interp_spect(self.df,*args,**kwargs))
    
    def mask(self,*args,**kwargs):
        return spectral_data(mask(self.df,*args,**kwargs))
        
    def random_folds(self,*args,**kwargs):
        return spectral_data(random(self.df,*args,**kwargs))
        
    def norm(self,*args,**kwargs):
        return spectral_data(norm_spect(self.df,*args,**kwargs))
        
    def remove_baseline(self,*args,**kwargs):
        df,baseline=remove_baseline(self.df,*args,**kwargs)                
        return spectral_data(df),spectral_data(baseline)
        
        