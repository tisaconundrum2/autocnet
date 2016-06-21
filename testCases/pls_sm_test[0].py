__author__ = 'nfinch'
__author__ = 'rbanderson'


###################################################
# import all preprocessors
###################################################

import pandas as pd
import numpy as np
from autocnet.spectral.spectral_data import spectral_data
from autocnet.regression.pls_sm import pls_sm
import matplotlib.pyplot as plot
import os

###################################################
# General setup
# we'll use the os class to find current user
# we'll have known and unknown databases and mask
###################################################

# Known Data
known_db = os.path.expanduser("~\\full_db_mars_corrected_dopedTiO2_pandas_format.csv")
panda_known_db = pd.read_csv(known_db, header=[0, 1])
# Unknown Data
unknown_db = os.path.expanduser("~\\lab_data_averages_pandas_format.csv")
panda_unknown_db = pd.read_csv(unknown_db, header=[0, 1])
# Mask File
mask_file = os.path.expanduser("~\\mask_minors_nouse.csv")

###################################################
# Spectral setup
# spectral analysis data
###################################################

# Known Data
k_spec_db = spectral_data(panda_known_db)
# Uknown Data
u_spec_db = spectral_data(panda_unknown_db)

###################################################
# Interpolate uknown spec data
###################################################
u_spec_db.interp(data.df['wv1'].columns)

###################################################
# Mask data
###################################################
data.mask(mask_file)
u_spec_db.mask(mask_file)


###################################################
# Normalizing Data and getting Ranges
###################################################

def get_range3(r0=0, r1=100, r2=300, r3=1000):
    return [(r0, r1), (r1, r2), (r2, r3)]

range3 = get_range_3(0, 350, 470, 1000)
range1 = [(0, 1000)]

###################################################
# nfolds: the number of folds to divide data into to extract an overall test set
# testfold: which fold to use as the overall test set
# nfoldsCV: Number of folds for CV
# testfoldCV: Which fold to use as the test set for cross validation
#
# Cross Validation
###################################################

element_name = "SiO2"
n_folds_test = 6
n_folds_cv = 5
test_fold_test = 4
test_fold_cv = 3

###################################################
# Composition Ranges
# These are the composition ranges for the submodels
###################################################

