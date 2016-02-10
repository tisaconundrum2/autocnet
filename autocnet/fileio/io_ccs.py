# This code is used to read individual ChemCam CCS .csv files
# Header data is stored as attributes of the data frame
# White space is stripped from the column names
import os
import numpy as np
import pandas as pd
import scipy
from autocnet.fileio.header_parser import header_parser
from autocnet.fileio.utils import file_search

def CCS(input_data):
    df = pd.DataFrame.from_csv(input_data, header=14)
    df.rename(columns=lambda x: x.strip(),inplace=True) #strip whitespace from column names
    df=df.transpose()
    
    cols=df.columns.tolist()
    for i,x in enumerate(cols):
        cols[i]=('wvl',round(float(x),5))
    df.columns=pd.MultiIndex.from_tuples(cols)
    #extract info from the file name
    fname=os.path.basename(input_data)
    df['sclock']=fname[4:13]
    df['sclock']=pd.to_numeric(df['sclock'])
    df['seqid']=fname[25:34].upper()
    df['Pversion']=fname[34:36]        
    #transpose the data frame
    
    #read the file header and put information into the dataframe as new columns (inneficient, but much easier to concatenate data from multiple files)
    with open(input_data,'r') as f:
        header={}
        for i,row in enumerate(f.readlines()):
            if i<14:
                row=row.split(',')[0]
                header.update(header_parser(row,'='))    
                
    for label,data in header.items(): 
        if '_float' in label:
            label=label.replace('_float','')
        df[label]=data 

    return df
        
def CCS_SAV(input_data):
    
    d=scipy.io.readsav(input_data,python_dict=True)
    #combine the three spectrometers
    spectra=np.vstack([d['uv'],d['vis'],d['vnir']])
    aspectra=np.array([np.hstack([d['auv'],d['avis'],d['avnir']])]).T
    mspectra=np.array([np.hstack([d['muv'],d['mvis'],d['mvnir']])]).T
    
    wvls=list(np.hstack([d['defuv'],d['defvis'],d['defvnir']]))
    for i,x in enumerate(wvls):
        wvls[i]=('wvl',round(x,5))
    
    #remove the above elements from the dict
    del d['uv']
    del d['vis']
    del d['vnir']
    del d['auv']
    del d['avis']
    del d['avnir']
    del d['muv']
    del d['mvis']
    del d['mvnir']
    del d['defuv']
    del d['defvis']
    del d['defvnir']
    
    #define column names
    shotnums=list(range(1,d['nshots']+1))
    shots=['shot'+str(i) for i in shotnums]
    shots.extend(['ave','median'])
    df = pd.DataFrame(np.hstack([spectra,aspectra,mspectra]),columns=shots,index=pd.MultiIndex.from_tuples(wvls))        
    df=df.transpose()

        #        #extract data from the PDS label info
#        pdslabel={}
#        for i in d['label_info']:
#            print(str(i.decode()))
#            if type(i) is bytes:
#                pdslabel.update(io_header_parser(i.decode(),'='))
#            elif len(i)>0:
#                pdslabel.update(io_header_parser(i,'='))
        
        
    del d['label_info']  #not currently using PDS label info        
    
    #extract info from the file name
    fname=os.path.basename(input_data)
    d['sclock']=fname[4:13]
    d['seqid']=fname[25:34].upper()
    d['Pversion']=fname[34:36]
    for label,data in d.items(): 
        if type(data) is bytes: data=data.decode()
        df[label]=data
    
    df['sclock']=pd.to_numeric(df['sclock'])
   
    
    return df    

def ccs_batch(directory,searchstring='*CCS*.csv',is_sav=False):
    if 'SAV' in searchstring:
        is_sav=True
    else:
        is_sav=False
    filelist=file_search(directory,searchstring)
    for i in filelist:
        
        if is_sav:
            tmp=CCS_SAV(i)
        else:
            tmp=CCS(i)
            
        try:
            cols1=list(combined.columns[combined.dtypes=='float'])
            cols2=list(tmp.columns[tmp.dtypes=='float'])
            if set(cols1)==set(cols2):
                combined=pd.concat([combined,tmp])
            else:
                print("Wavelengths don't match!")
                print('foo')
        except:
            combined=tmp
    return combined
    
        