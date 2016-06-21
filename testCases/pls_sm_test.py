# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:09:29 2016

@author: rbanderson
"""
import pandas as pd
import numpy as np
from autocnet.spectral.spectral_data import spectral_data
from autocnet.regression.pls_sm import pls_sm
#from autocnet.regression.pls_cv import pls_cv

import matplotlib.pyplot as plot

#Read training database
db=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Sample_Data\full_db_mars_corrected_dopedTiO2_pandas_format.csv"
data=pd.read_csv(db,header=[0,1])

######read unknown data (only do this the first time since it's slow)
#unknowndatadir=r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Lab Data"
#unknowndatasearch='CM*.SAV'
#unknowndatacsv=r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Lab Data\lab_data_averages_pandas_format.csv"
#unknown_data=ccs_batch(unknowndatadir,searchstring=unknowndatasearch)
#
##write it to a csv file for future use (much faster than reading individual files each time)
#
##this writes all the data, including single shots, to a file (can get very large!!)
#unknown_data.df.to_csv(unknowndatacsv)
#
##this writes just the average spectra to a file
#unknown_data.df.loc['average'].to_csv(unknowndatacsv)

#put the training data dataframe into a spectral_data object
data=spectral_data(data)


##########read unknown data from the combined csv file (much faster)
unknowndatacsv=r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\Lab Data\lab_data_averages_pandas_format.csv"
unknown_data=pd.read_csv(unknowndatacsv,header=[0,1])
unknown_data=spectral_data(unknown_data)

#Interpolate unknown data onto the same exact wavelengths as the training data
unknown_data.interp(data.df['wvl'].columns)

#Mask out unwanted portions of the data
maskfile=r"C:\Users\rbanderson\Documents\Projects\LIBS PDART\Input\mask_minors_noise.csv"
data.mask(maskfile)
unknown_data.mask(maskfile)


# TODO Create a saving feature for the user
# TODO creaCreate slidings bars, with numbers adjustable numbers
# TODO Create submodel sliding bars, with adjustable numbers for Submodel
# TODO reate a way to make steps customizable. Ordering of functions
# TODO first task, understand the code :D


# int range = a certain number
# list [] = use sliding bar for this.
# range_of_two_values = low, high;
# range 3
# <-------------------|---------------------|--------------------->
# 0                   400                   500                  1000
# <-----------|-----------------------------|--------------------->
# []          250                            500                  []
# [] = numbers that users can add themselves


# range of 3, create 3 sliding bars
# |-------|--------|
# 0      250      400
# |-------|--------|
# 250
# |-------|--------|


#Normalize spectra by specifying the wavelength ranges over which to normalize
ranges3=[(0,350),(350,470),(470,1000)] #this is equivalent to "norm3"
#create a field for the user to work with, make sliding bars move according to what the user enters
ranges1=[(0,1000)] #this is equivalent to "norm1"

#Norm3 data
data3=data
data3.norm(ranges3)
unknown_data3=unknown_data
unknown_data3.norm(ranges3)

#norm1 data
data1=data
data1.norm(ranges1)
unknown_data1=unknown_data
unknown_data1.norm(ranges1)



#set up for cross validation
el='SiO2'
nfolds_test=6 #number of folds to divide data into to extract an overall test set
testfold_test=4 #which fold to use as the overall test set
nfolds_cv=5  #number of folds to use for CV
testfold_cv=3 #which fold to use as test set for the cross validation models

compranges=[[-20,50],[30,70],[60,100],[0,120]] #these are the composition ranges for the submodels
nc=20  #max number of components
outpath=r'C:\Users\rbanderson\Documents\Projects\LIBS PDART\Output'

#remove a test set to be completely excluded from CV and used to assess the final blended model
data3.stratified_folds(nfolds=nfolds_test,sortby=('meta',el))
data3_train=data3.rows_match(('meta','Folds'),[testfold_test],invert=True)
data3_test=data3.rows_match(('meta','Folds'),[testfold_test])
 
data1.stratified_folds(nfolds=nfolds_test,sortby=('meta',el))
data1_train=data1.rows_match(('meta','Folds'),[testfold_test],invert=True)
data1_test=data1.rows_match(('meta','Folds'),[testfold_test])
   

#do cross validation for each compositional range
#If you know how many components you want to use for each submodel, you can comment this loop out
#for n in compranges:
#    #First use the norm1 data
#    data1_tmp=within_range(data1_train,n,el)
#    #Split the known data into stratified train/test sets for the element desired
#    data1_tmp=folds.stratified(data1_tmp,nfolds=nfolds_cv,sortby=('meta',el))
#    #Separate out the train and test data
#    train_cv=data1_tmp.loc[-data1_tmp[('meta','Folds')].isin([testfold_cv])]
#    test_cv=data1_tmp.loc[data1_tmp[('meta','Folds')].isin([testfold_cv])]
#
#    figfile='PLS_CV_nc'+str(nc)+'_'+el+'_'+str(n[0])+'-'+str(n[1])+'_norm1.png'
#    norm1_rmses=pls_cv(train_cv,Test=test_cv,nc=nc,nfolds=nfolds_cv,ycol=el,doplot=True,
#           outpath=outpath,plotfile=figfile)
#
#    #next use the norm3 data
#    data3_tmp=within_range(data3_train,n,el)
#    #Split the known data into stratified train/test sets for the element desired
#    data3_tmp=folds.stratified(data3_tmp,nfolds=nfolds_cv,sortby=('meta',el))
#    #Separate out the train and test data
#    train_cv=data3_tmp.loc[-data3_tmp[('meta','Folds')].isin([testfold_cv])]
#    test_cv=data3_tmp.loc[data3_tmp[('meta','Folds')].isin([testfold_cv])]
#
#    figfile='PLS_CV_nc'+str(nc)+'_'+el+'_'+str(n[0])+'-'+str(n[1])+'_norm3.png'
#    norm3_rmses=pls_cv(train_cv,Test=test_cv,nc=nc,nfolds=nfolds_cv,ycol=el,doplot=True,
#           outpath=outpath,plotfile=figfile)
#

#At this point, stop and look at the plots produced by cross validation and 
#use them to choose the number of components for each of the submodels. 

#Eventually I will add options to automatically define the "best" number of components
#but for now it is still human-in-the-loop


#Here the models are in the order: Low, Mid, High, Full
#(The "full" model, which is being used as the reference to determine which submodel is appropriate
#should always be the last one)
ncs=[7,7,5,9] #provide the best # of components for each submodel
traindata=[data3_train.df,data3_train.df,data1_train.df,data3_train.df] #provide training data for each submodel
testdata=[data3_test.df,data3_test.df,data1_test.df,data3_test.df] #provide test data for each submodel
unkdata=[unknown_data3.df,unknown_data3.df,unknown_data1.df,unknown_data3.df] #provide unknown data to be fed to each submodel

#create an instance of the submodel object
sm=pls_sm()

#Fit the submodels on the training data
#outpath specifies where to write the outlier check plots
sm.fit(traindata,compranges,ncs,el,figpath=outpath)

#predict the training data
predictions_train=sm.predict(traindata)
#predict the test data
predictions_test=sm.predict(testdata)
#predict the unknown data
#predictions_unk=sm.predict(unkdata)

#Combine the submodel predictions for the training data, optimizing the blending ranges in the process
blended_train=sm.do_blend(predictions_train,traindata[0]['meta'][el])
#Combine the predictions for the test data
blended_test=sm.do_blend(predictions_test)
#combine the predictions for the unknown data
#blended_unk=sm.do_blend(predictions_unk)
#put them in a data frame



outpath=r'C:\Users\rbanderson\Documents\Projects\LIBS PDART\Output'


#Make a figure showing the test set performance
plot.figure()
plot.scatter(testdata[0]['meta'][el],blended_test,color='r')
plot.plot([0,100],[0,100])
plot.show()

print(foo)
    





