# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:25:47 2016

@author: rbanderson
"""
#import sys
#sys.path.append(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART")
from autocnet.fileio.io_ccs import ccs_batch
from autocnet.fileio.io_jsc import JSC,jsc_batch,read_refdata
from autocnet.fileio.lookup import lookup
from autocnet.spectral.interp import interp_spect
from autocnet.spectral.mask import mask
from autocnet.spectral.spectra import Spectra
from autocnet.spectral.spectral_data import spectral_data
from autocnet.spectral.norm_total import norm_total,norm_spect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import time
from sklearn.decomposition import PCA


##Read CCAM data
#data_dir=r"E:\ChemCam\ops_ccam_team\sav\0-250"
#
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs=ccs_batch(data_dir,searchstring='*CCS*.SAV')
#dt1=time.time()-t1
#
##work only with average spectra
#ccs=ccs.loc[ccs['shotnum'].isin(['ave'])]
#ccs=ccs.reset_index(drop=True)  #This is important! without it, the lookup is screwed up
#ccs=lookup(ccs,masterlists)
#
##save ccs data
#ccs.to_csv('CCAM_data_aves_0-250.csv')
#
#data_dir=r"E:\ChemCam\ops_ccam_team\sav\251-500"
#
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs=ccs_batch(data_dir,searchstring='*CCS*.SAV')
#dt2=time.time()-t1
#
##work only with average spectra
#ccs=ccs.loc[ccs['shotnum'].isin(['ave'])]
#ccs=ccs.reset_index(drop=True)  #This is important! without it, the lookup is screwed up
#ccs=lookup(ccs,masterlists)
#
##save ccs data
#ccs.to_csv('CCAM_data_aves_251-500.csv')
#
#data_dir=r"E:\ChemCam\ops_ccam_team\sav\501-750"
#
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs=ccs_batch(data_dir,searchstring='*CCS*.SAV')
#dt3=time.time()-t1
#
##work only with average spectra
#ccs=ccs.loc[ccs['shotnum'].isin(['ave'])]
#ccs=ccs.reset_index(drop=True)  #This is important! without it, the lookup is screwed up
#ccs=lookup(ccs,masterlists)
#
##save ccs data
#ccs.to_csv('CCAM_data_aves_501-750.csv')
#
#data_dir=r"E:\ChemCam\ops_ccam_team\sav\751-1000"
#
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs=ccs_batch(data_dir,searchstring='*CCS*.SAV')
#dt4=time.time()-t1
#
##work only with average spectra
#ccs=ccs.loc[ccs['shotnum'].isin(['ave'])]
#ccs=ccs.reset_index(drop=True)  #This is important! without it, the lookup is screwed up
#ccs=lookup(ccs,masterlists)
#
##save ccs data
#ccs.to_csv('CCAM_data_aves_751-1000.csv')
#
#data_dir=r"E:\ChemCam\ops_ccam_team\sav\1001-1250"
#
#masterlists=[r"E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0010_0801.csv","E:\ChemCam\ops_ccam_misc\MASTERLIST_SOL_0805_0980.csv",r"E:\ChemCam\ops_ccam_misc\MASTERLIST.csv"]
#t1=time.time()
#ccs=ccs_batch(data_dir,searchstring='*CCS*.SAV')
#dt5=time.time()-t1
#
##work only with average spectra
#ccs=ccs.loc[ccs['shotnum'].isin(['ave'])]
#ccs=ccs.reset_index(drop=True)  #This is important! without it, the lookup is screwed up
#ccs=lookup(ccs,masterlists)
#
##save ccs data
#ccs.to_csv('CCAM_data_aves_1001_1250.csv')

#f1=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_0-250.csv"
#f2=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_251-500.csv"
#f3=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_501-750.csv"
#f4=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_751-1000.csv"
#f5=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_1001_1250.csv"
#
#ccs1=pd.read_csv(f1,header=[0,1])
#ccs2=pd.read_csv(f2,header=[0,1])
#ccs3=pd.read_csv(f3,header=[0,1])
#ccs4=pd.read_csv(f4,header=[0,1])
#ccs5=pd.read_csv(f5,header=[0,1])
#
#ccs=pd.concat([ccs1,ccs2,ccs3,ccs4,ccs5])
####
#ccs.to_csv('CCAM_data_aves.csv')
ccs=pd.read_csv(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves.csv",header=[0,1])
pca=PCA(n_components=2)
ccs_geo=ccs.loc[ccs['meta']['Distance (mm)']>1.7]

##Filter out just Ti targets
#ccs_Ti=ccs.loc[np.squeeze(ccs['meta']['Target'].isin(['Cal Target 10']))]
#ccs_Ti.to_csv('CCAM_data_aves_Ti.csv')

ccs_Ti=pd.read_csv(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\CCAM_data_aves_Ti.csv",header=[0,1])


xnew=np.array(ccs_Ti['wvl'].columns)
ccs_Ti=interp_spect(ccs_Ti,xnew)
ccs_geo=interp_spect(ccs_geo,xnew)

plot.figure(figsize=(10,8))
plot.subplot(311)
plot.xlim([200,900])
rocknest3=ccs_geo.loc[ccs['meta']['Target'].isin(['Rocknest3'])]
plot.plot(rocknest3['wvl'].columns.values,rocknest3['wvl'].iloc[0,:],label='Raw',c='b')    
plot.legend()

#Mask spectra
maskfile=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\mask_minors_noise.csv"
ccs_Ti=mask(ccs_Ti,maskfile)
ccs_geo=mask(ccs_geo,maskfile)
plot.subplot(312)
plot.xlim([200,900])
rocknest3=ccs_geo.loc[ccs['meta']['Target'].isin(['Rocknest3'])]
plot.plot(rocknest3['wvl'].columns.values,rocknest3['wvl'].iloc[0,:],label='Masked',c='r')    
plot.legend()
#Normalize Spectra
ranges=[(0,350),(350,460),(460,1000)]
ccs_Ti=norm_spect(ccs_Ti,ranges)
ccs_geo=norm_spect(ccs_geo,ranges)
plot.subplot(313)
plot.xlim([200,900])
rocknest3=ccs_geo.loc[ccs['meta']['Target'].isin(['Rocknest3'])]
plot.plot(rocknest3['wvl'].columns.values,rocknest3['wvl'].iloc[0,:],label='Normalized',c='g')    

plot.legend()
plot.savefig('Rocknest_example.png',dpi=600)
plot.show()

do_pca=pca.fit(ccs_geo['wvl'])
seqs=ccs_geo['meta']['Sequence']
seqs_uniq=np.unique(seqs)
plot.figure(figsize=(8,8))
plot.title('PCA of Mars Targets')
plot.xlabel('PC1 ('+str(round(do_pca.explained_variance_ratio_[0],2))+'%)')
plot.ylabel('PC2 ('+str(round(do_pca.explained_variance_ratio_[1],2))+'%)')

colors=plot.cm.jet(np.linspace(0,1,len(seqs_uniq)))
for t,i in enumerate(seqs_uniq):
    
    scores=do_pca.transform(ccs_geo['wvl'].loc[ccs_geo['meta']['Sequence'].isin([i])])
    plot.scatter(scores[:,0],scores[:,1],c=colors[t,:],label=i)
plot.savefig('Full_CCS_PCA.png',dpi=600)    
plot.show()

pca=PCA(n_components=2)
do_pca=pca.fit(ccs_Ti['wvl'])
scores_ccs_Ti=do_pca.transform(ccs_Ti['wvl'])


plot.figure()
plot.scatter(scores_ccs_Ti[:,0],scores_ccs_Ti[:,1],c='r')
plot.show()

ccs_Ti=ccs_Ti.iloc[scores_ccs_Ti[:,0]<0.06,:]
do_pca=pca.fit(ccs_Ti['wvl'])
scores_ccs_Ti=do_pca.transform(ccs_Ti['wvl'])


plot.figure()
plot.scatter(scores_ccs_Ti[:,0],scores_ccs_Ti[:,1],c='r')
plot.show()


#get average mars spectra
ccs_Ti_ave=ccs_Ti['wvl'].sum(axis=0)/len(ccs_Ti.index)


#Read JSC data
#spect_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Spectrometer_Table.csv"
#experiment_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Experiment_Setup_Table.csv"
#laser_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Laser_Setup_Table.csv"
#sample_table=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\Sample_Table.csv"
#LUT_files={'spect':spect_table,'exp':experiment_table,'laser':laser_table,'sample':sample_table}
#data_dir=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\LIBS USGS\DATA"
#JSC_data=jsc_batch(data_dir,LUT_files)
#JSC_data.to_csv('JSC_data.csv')
##Filter out just the Ti targets
#JSC_Ti=JSC_data.loc[np.squeeze(JSC_data['Sample ID'].isin(['TISDT01']))]

JSC_Ti=pd.read_csv(r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\autocnet\JSC_Ti_data.csv",header=[0,1])

#Interpolate JSC data to CCAM data
JSC_Ti=interp_spect(JSC_Ti,xnew)

##Combine JSC and CCAM Ti data
#data=pd.concat([JSC_Ti_interp,ccs_Ti])
#data.to_csv('JSC_CCS_Ti_data.csv')
#Mask spectra
JSC_Ti=mask(JSC_Ti,maskfile)
#Normalize Spectra

JSC_Ti=norm_spect(JSC_Ti,ranges)
#data_masked['wvl']=norm_total(data_masked['wvl'])
#
#data_masked['wvl']=data_masked['wvl'].div(data_masked['wvl'].sum(axis=1),axis=0)
#
#data_mask_norm=data_masked['wvl'].copy()
#for row in data_mask_norm.index.values:
#    data_mask_norm.iloc[row]/=sum(data_mask_norm.iloc[row])
#data_masked['wvl']=data_mask_norm
#data_masked_norm.to_csv('JSC_CCS_Ti_data_masked_norm.csv')    
#data_mask_norm=norm_total(data_masked)
#data_mask_norm.to_csv('JSC_CCS_Ti_data_mask_norm.csv')
#print('foo')

#get average of JSC spectra
JSC_ave=JSC_Ti['wvl'].sum(axis=0)/len(JSC_Ti.index)

ratio=ccs_Ti_ave/JSC_ave
ratio[abs(ratio)>100]=1.0
plot.plot(ratio)
plot.show()

JSC_Ti_r=JSC_Ti['wvl'].mul(ratio,axis=1)
JSC_Ti_1248=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([12.48])]
JSC_Ti_1196=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([11.98])]
JSC_Ti_1498=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([14.98])]
JSC_Ti_1723=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([17.23])]

JSC_Ti_1248_ave=JSC_Ti_1248['wvl'].sum(axis=0)/len(JSC_Ti_1248.index)
JSC_Ti_1196_ave=JSC_Ti_1196['wvl'].sum(axis=0)/len(JSC_Ti_1196.index)
JSC_Ti_1498_ave=JSC_Ti_1498['wvl'].sum(axis=0)/len(JSC_Ti_1498.index)
JSC_Ti_1723_ave=JSC_Ti_1723['wvl'].sum(axis=0)/len(JSC_Ti_1723.index)

dist_1248=np.linalg.norm(JSC_Ti_1248_ave-ccs_Ti_ave)
dist_1196=np.linalg.norm(JSC_Ti_1196_ave-ccs_Ti_ave)
dist_1498=np.linalg.norm(JSC_Ti_1498_ave-ccs_Ti_ave)
dist_1723=np.linalg.norm(JSC_Ti_1723_ave-ccs_Ti_ave)

#combine mars and JSC data
data=pd.concat([JSC_Ti_r,ccs_Ti['wvl']])


#Run PCA on spectra
pca=PCA(n_components=2)
do_pca=pca.fit(data)
scores_all=do_pca.transform(data)



#Extract different laser energies
mars_40A=ccs_Ti.loc[ccs_Ti['meta']['Laser Energy'].isin(['100A/40A/40A'])]['wvl']
mars_60A=ccs_Ti.loc[ccs_Ti['meta']['Laser Energy'].isin(['100A/60A/60A'])]['wvl']
mars_95A=ccs_Ti.loc[ccs_Ti['meta']['Laser Energy'].isin(['100A/95A/95A'])]['wvl']


JSC_1248=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([12.48])]['wvl'].mul(ratio,axis=1)
JSC_1196=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([11.96])]['wvl'].mul(ratio,axis=1)
JSC_1498=JSC_Ti.loc[JSC_Ti['meta']['laser_power'].isin([14.98])]['wvl'].mul(ratio,axis=1)


scores_40A=do_pca.transform(mars_40A)
scores_60A=do_pca.transform(mars_60A)
scores_95A=do_pca.transform(mars_95A)

scores_1248=do_pca.transform(JSC_1248)
scores_1196=do_pca.transform(JSC_1196)
scores_1498=do_pca.transform(JSC_1498)

plot.figure(figsize=(5,5))
plot.scatter(scores_40A[:,0],scores_40A[:,1],label='Mars (40A)',c='r')
plot.scatter(scores_60A[:,0],scores_60A[:,1],label='Mars (60A)',c='g')
plot.scatter(scores_95A[:,0],scores_95A[:,1],label='Mars (95A)',c='b')

plot.scatter(scores_1248[:,0],scores_1248[:,1],label='JSC (12.48 mJ)',c='c')
plot.scatter(scores_1196[:,0],scores_1196[:,1],label='JSC (11.96 mJ)',c='m')
plot.scatter(scores_1498[:,0],scores_1498[:,1],label='JSC (14.98 mJ)',c='y')
plot.legend()

plot.savefig('PCA_Ti_JSC_CCS.png',dpi=600)
plot.show()
print('foo')

