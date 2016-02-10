# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:55:46 2015

@author: rbanderson
"""
import autocnet
from autocnet.fileio.io_ccs import CCS,CCS_SAV,ccs_batch
from autocnet.fileio.io_jsc import JSC,jsc_batch,read_refdata
from autocnet.fileio.io_edr import EDR
from autocnet.fileio.io_csv import CSV
from autocnet.fileio.lookup import lookup
from autocnet.spectral.interp import interp_spect
from autocnet.spectral.mask import mask
from autocnet.spectral.spectra import Spectra
from autocnet.spectral.spectral_data import spectral_data
import pandas as pd

print("Test reading Chemcam CCS data")
data_file=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM\CL5_398645626CCS_F0030004CCAM02013P1.csv"
ccs_result=CCS(data_file)
ccs=spectral_data(ccs_result)

data_file=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM\CL5_398645626CCS_F0030004CCAM02013P1.SAV"
ccs_sav_result=CCS_SAV(data_file)
#ccs_sav_result=spectral_data(ccs_sav_result)

data_dir=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM"
ccs_batch_csv=ccs_batch(data_dir,searchstring='*CCS*.csv')
ccs_batch_SAV=ccs_batch(data_dir,searchstring='*CCS*.SAV')

print("Test reading JSC data")
spect_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Spectrometer_Table.csv"
experiment_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Experiment_Setup_Table.csv"
laser_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Laser_Setup_Table.csv"
sample_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Sample_Table.csv"
LUT_files={'spect':spect_table,'exp':experiment_table,'laser':laser_table,'sample':sample_table}
refdata=read_refdata(LUT_files)

data_dir=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\LIBS USGS\DATA"
JSC_data=jsc_batch(data_dir,LUT_files)
data_file=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\LIBS USGS\DATA\LIB00001_02_J_B7.29_A12.48_A_S594_10-49-43-063.txt"
JSC_single=JSC(data_file,refdata)


print("Test reading data from CSV with lots of spectra and their metadata in one file")
dbfile=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\full_db_mars_corrected_dopedTiO2.csv"
db=CSV(dbfile,setindex='Name')

print("Test assigning random folds")
ccs=spectral_data(ccs_batch_csv)
ccs.random_folds(nfolds=6,seed=1,groupby='seqid')

print("Test looking up ChemCam metadata")

masterlist_files=[r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM\MASTERLIST.csv",
                  r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM\MASTERLIST_SOL_0010_0801.csv",
                  r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM\MASTERLIST_SOL_0805_0980.csv"]

ccs=lookup(ccs.df,masterlist_files)
#
#jsctest=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\LIBS_TEST\TestSS_UV_01.txt"
#jsc=JSC(jsctest)

#foo=db.interp_spectra(db,newx)
#
#edrtest=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\pysat\pysat\examples\ChemCam\CL5_399178818EDR_F0030078CCAM01019M1_spect.TXT"
#edr=EDR(edrtest)
#


#maskfile=r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\Input\mask_minors_noise.csv"
#masksav=sav.mask(sav,maskfile)

print("foo")