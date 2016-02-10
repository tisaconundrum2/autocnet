# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:55:46 2015

@author: rbanderson
"""
import autocnet
from autocnet.fileio.io_ccs import CCS,CCS_SAV,ccs_batch
from autocnet.fileio.io_jsc import JSC,jsc_batch
from autocnet.fileio.io_edr import EDR
from autocnet.fileio.io_csv import CSV
from autocnet.fileio.lookup import lookup
from autocnet.spectral.interp import interp_spect
from autocnet.spectral.mask import mask
from autocnet.spectral.spectra import Spectra
from autocnet.spectral.spectral_data import spectral_data
import pandas as pd

#
#df1=pd.DataFrame(['a','b',1.0,5.6,7.6,6.8],index=['A','B','C',1.0,2.0,3.0]).T
#df2=pd.DataFrame(['a','b',1.0,5.6,7.6,6.8],index=['A','B','C',4.0,5.0,6.0]).T
#df3=pd.DataFrame(['a','b',1.0,5.6,7.6,6.8],index=['A','B','C',7.0,8.0,9.0]).T
#df4=pd.DataFrame(['a1','b',1.0,5.6,7.6,6.8],index=['A','B','C',1.0,2.0,3.0]).T
#df5=pd.DataFrame(['a1','b',1.0,5.6,7.6,6.8],index=['A','B','C',4.0,5.0,6.0]).T
#df6=pd.DataFrame(['a1','b',1.0,5.6,7.6,6.8],index=['A','B','C',7.0,8.0,9.0]).T
#







spect_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Spectrometer_Table.csv"
experiment_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Experiment_Setup_Table.csv"
laser_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Laser_Setup_Table.csv"
sample_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Sample_Table.csv"
LUT_files={'spect':spect_table,'exp':experiment_table,'laser':laser_table,'sample':sample_table}

data_dir=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\LIBS USGS\DATA"
foo=jsc_batch(data_dir,LUT_files)
testdata=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam\CL5_398645626CCS_F0030004CCAM02013P3.csv"
ccs_result=CCS(testdata)
ccs=spectral_data(ccs_result)
ccs_br=ccs_batch(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam",searchstring='*CCS*.csv')
ccs_br2=ccs_batch(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam",searchstring='*CCS*.SAV')
ccs_br=spectral_data(ccs_br)
ccs_br.random_folds(nfolds=6,seed=1,groupby='seqid')

foo=jsc_batch(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\LIBS_TEST",searchstring='*.txt')
foo.transpose().sort_index(level=1).to_csv('JSC_output_test.csv')

newx=list(foo.wvl.columns)
blah=ccs.interp(newx)

dbfile=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\full_db_mars_corrected_dopedTiO2.csv"
db=CSV(dbfile,setindex='Name')

print('foo')
#
#
##masterlist=["E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
##blah=lookup(ccs_br,masterlist)
#
#jsctest=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\LIBS_TEST\TestSS_UV_01.txt"
#jsc=JSC(jsctest)

foo=db.interp_spectra(db,newx)
#
#edrtest=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam\CL5_399178818EDR_F0030078CCAM01019M1_spect.TXT"
#edr=EDR(edrtest)
#

savtest=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam\CL5_398736801CCS_F0030004CCAM01014P3.SAV"
sav=CCS_SAV(savtest)
maskfile=r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\Input\mask_minors_noise.csv"
masksav=sav.mask(sav,maskfile)

print("foo")