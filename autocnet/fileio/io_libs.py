import datetime
import os
import re

import numpy as np
import pandas as pd

from autocnet.spectral.spectra import Spectra

class LIBS(object):
    
    def __init__(self, input_data):
        self.spectra = None
        with open(input_data, 'r') as f:
            """
            Could easily add regex to the parsing to be more robust reading,
            could also peg metadata to t
            """

            for i, l in enumerate(f.readlines()):
                if i == 14:
                    wavelengths = np.fromstring(l, sep=' ')
                elif i > 14:
                    sl = l.split('\t')
                    time = sl[0]
                    sid = sl[1]
                    rawsp = np.asarray(map(float,sl[2:]))
                    if not self.spectra:
                        df = pd.DataFrame(rawsp, columns=[sid],
                                          index=wavelengths)
                        self.spectra = Spectra(df)
                    else:
                        self.spectra.df[sid] = rawsp
                elif i == 0 or i == 13:
                    pass
                elif i == 1:
                    date = ' '.join(l.rstrip().split(':')[1:])
                    #date = datetime.datetime(date) #Format needs to be specified
                    setattr(self, 'Date', date)
                else:
                    key, v = l.split(':')
                    k = '_'.join(key.split())
                    setattr(self, k, v.rstrip())
