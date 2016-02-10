# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:53:23 2015

@author: rbanderson
"""
from pysat.spectral.interp import interp_spect
from pysat.spectral.mask import mask
from pysat.utils.folds import random

class spectral_data(object):
    def __init__(self,df):
        self.df=df
    
    def interp(self,*args,**kwargs):
        return interp_spect(self.df,*args,**kwargs)
    
    def mask(self,*args,**kwargs):
        return mask(self.df,*args,**kwargs)
        
    def random_folds(self,*args,**kwargs):
        return random(self.df,*args,**kwargs)
        
        
        