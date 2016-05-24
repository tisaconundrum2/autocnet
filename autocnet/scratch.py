# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:03:23 2016

@author: rbanderson
"""
from autocnet.fileio.io_ccs import ccs_batch
from autocnet.fileio.io_jsc import jsc_batch
from autocnet.spectral.spectral_data import spectral_data
import pandas as pd
import matplotlib.pyplot as plot
import time
####Read CCAM data
#data_dir1=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\CCAM"
#data_dir2=r"E:\ChemCam\ops_ccam_team\csv\test"
##
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs1=ccs_batch(data_dir1,searchstring='*CCS*.SAV',to_csv=r'..\SAV_output_test.csv',lookupfile=masterlists)
#ranges=[[0,350],[360,470],[475,1000]]
#dt1=time.time()-t1
#
#foo=ccs1.norm(ranges)
#t2=time.time()
#ccs2=ccs_batch(data_dir2,searchstring='*CCS*.csv',to_csv=r'..\csv_output_test.csv',lookupfile=masterlists)
#dt2=time.time()-t2
#print(dt1)
#
#spect_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Spectrometer_Table.csv"
#experiment_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Experiment_Setup_Table.csv"
#laser_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Laser_Setup_Table.csv"
#sample_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Sample_Table.csv"
#LUT_files={'spect':spect_table,'exp':experiment_table,'laser':laser_table,'sample':sample_table}
#
#data_dir=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\LIBS USGS\DATA"
#t=time.time()
#JSC_data=jsc_batch(data_dir,LUT_files,to_csv='../../JSC_test.csv')
JSC=spectral_data(pd.read_csv('../../JSC_test.csv',header=[0,1],index_col=0))
JSC_als,als_baseline=JSC.remove_baseline(method='als')
JSC_dietrich,dietrich_baseline=JSC.remove_baseline(method='dietrich')
JSC_polyfit,polyfit_baseline=JSC.remove_baseline(method='polyfit')
JSC_airpls,airpls_baseline=JSC.remove_baseline(method='airpls')
JSC_fabc,fabc_baseline=JSC.remove_baseline(method='fabc')
JSC_kk,kk_baseline=JSC.remove_baseline(method='kk')
#JSC_mario,mario_baseline=JSC.remove_baseline(method='mario')
JSC_median,median_baseline=JSC.remove_baseline(method='median')
JSC_rubberband,rubberband_baseline=JSC.remove_baseline(method='rubberband')

wvls=JSC.df['wvl'].columns.values
plot.plot(wvls,JSC.df['wvl'].loc[0,:])
plot.plot(wvls,als_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,dietrich_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,polyfit_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,airpls_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,fabc_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,kk_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,median_baseline.df['wvl'].loc[0,:])
plot.plot(wvls,rubberband_baseline.df['wvl'].loc[0,:])
plot.show()
#JSC_wavelet,wavelet_baseline=JSC.remove_baseline(method='wavelet')



dt=time.time()-t
print(dt)