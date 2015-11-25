import os
import re

import numpy as np
import pandas as pd

from pysat.spectral.spectra import Spectra

class RMI(object):
    def __init__(self, input_data):
        df = pd.DataFrame.from_csv(input_data, header=14)
        self.spectra = Spectra(df)

