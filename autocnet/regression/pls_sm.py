# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:15:46 2016

@author: rbanderson
"""
import numpy as np
import autocnet.spectral.within_range as within_range
from sklearn.cross_decomposition.pls_ import PLSRegression
from autocnet.spectral.meancenter import meancenter
from matplotlib import pyplot as plot
import scipy.optimize as opt
    
class pls_sm:
    def __init__(self):
        pass
    
    def fit(self,trainsets,ranges,ncs,ycol,figpath=None):
        self.ranges=ranges
        self.ncs=ncs        
        self.ycol=ycol
        submodels=[]    
        mean_vects=[]
        for i,rangei in enumerate(ranges):
            data_tmp=within_range.within_range(trainsets[i],rangei,ycol)
            x=data_tmp.xs('wvl',axis=1,level=0,drop_level=False)
            y=data_tmp['meta'][ycol]
            x_centered,x_mean_vect=meancenter(x) #mean center training data
            pls=PLSRegression(n_components=ncs[i],scale=False)
            pls.fit(x,y)
            submodels.append(pls)
            mean_vects.append(x_mean_vect)
            if figpath is not None:
                E=x_centered-np.dot(pls.x_scores_,pls.x_loadings_.transpose())
                Q_res=np.dot(E,E.transpose()).diagonal()
                T=pls.x_scores_
                #There's probably a more efficient way to calculate T2...
                leverage=np.zeros_like(Q_res)
                for k in range(len(Q_res)):
                    leverage[k]=np.dot(T[k],np.dot(np.linalg.inv(np.dot(T.transpose(),T)),T[k]))
                plot.figure()
                plot.scatter(leverage,Q_res,color='r',edgecolor='k')
                plot.title(ycol+' ('+str(rangei[0])+'-'+str(rangei[1])+')')
                plot.xlabel('Leverage')
                plot.ylabel('Q')
                    
                plot.savefig(figpath+'/'+ycol+'_'+str(rangei[0])+'-'+str(rangei[1])+'Qres_vs_Leverage.png',dpi=600)
                self.leverage=leverage
                self.Q_res=Q_res
            self.submodels=submodels
            self.mean_vects=mean_vects
         
    def do_blend(self,predictions,truevals=None):
        
        
        #get the ranges that are not the reference model (assumed to be the last model)
        ranges_sub=self.ranges[:-1]
        blendranges=np.array(ranges_sub).flatten() #squash them to be a 1d array
        blendranges.sort()  #sort the entries 
         
        #create the array indicating which models to blend for each blend range
        self.toblend=[]
        for i in range(len(predictions)-1):
            self.toblend.append([i,i])
            if i<len(predictions)-2:
                self.toblend.append([i,i+1])
            
        
        if truevals is not None:
            print('Optimizing blending ranges')
            result=opt.minimize(self.get_rmse,blendranges,(predictions,truevals))
            self.blendranges=result.x
        else:
            self.blendranges=blendranges
            
        blended=self.submodels_blend(predictions,result.x,overwrite=False,noneg=False)
        return blended
        
    def get_rmse(self,blendranges,predictions,truevals):
        print(blendranges)
        blended=self.submodels_blend(predictions,blendranges,overwrite=False,noneg=False)
        RMSE=np.sqrt(np.mean((blended-truevals)**2))
        return RMSE
        
    def submodels_blend(self,predictions,blendranges,overwrite=False,noneg=True):
        blended=np.squeeze(np.zeros_like(predictions[0]))
        blendranges=np.hstack((blendranges,blendranges[1:-1])) #duplicate the middle entries
        blendranges.sort() #re-sort them
        blendranges=np.reshape(blendranges,(len(blendranges)/2,2))  #turn the vector into a 2d array
        for i in range(len(blendranges)): #loop over each composition range
            for j in range(len(predictions[0])): #loop over each spectrum
                ref_tmp=predictions[-1][j]   #get the reference model predicted value
                #check whether the prediction for the reference spectrum is within the current range            
                inrangecheck=(ref_tmp>blendranges[i][0])&(ref_tmp<blendranges[i][1])
     
                if inrangecheck: 
                    if self.toblend[i][0]==self.toblend[i][1]: #if the results being blended are identical, no blending necessary!
                      
                        blendval=predictions[self.toblend[i][0]][j]
                    else:
                        weight1=1-(ref_tmp-blendranges[i][0])/(blendranges[i][1]-blendranges[i][0]) #define the weight applied to the lower model
                        weight2=(ref_tmp-blendranges[i][0])/(blendranges[i][1]-blendranges[i][0]) #define the weight applied to the higher model
                        blendval=weight1*predictions[self.toblend[i][0]][j]+weight2*predictions[self.toblend[i][1]][j] #calculated the blended value (weighted sum)
                    if overwrite:
                        blended[j]=blendval #If overwrite is true, write the blended result no matter what
                    else:
                        if blended[j]==0:  #If overwrite is false, only write the blended result if there is not already a result there
                            blended[j]=blendval                
        #Set any negative results to zero if noneg is true
        if np.min(blended)<0 and noneg==True:
            blended[blended<0]=0
    
        return blended
    
    def predict(self,x):
        #x is a list of data frames to feed into each submodel. 
        #This allows different normalizations to be used with each submodel
        predictions=[]
        for i,k in enumerate(self.submodels):
            xtemp=x[i].xs('wvl',axis=1,level=0,drop_level=False)
            xtemp,mean_vect=meancenter(xtemp,previous_mean=self.mean_vects[i])
            predictions.append(k.predict(xtemp['wvl']))
        return predictions
