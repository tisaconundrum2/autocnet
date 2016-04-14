import shutil
import tempfile
import os
import fnmatch
import numpy as np
import pandas as pd

def create_dir(basedir=''):
    """
    Create a unique, temporary directory in /tmp where processing will occur

    Parameters
    ----------
    basedir : str
              The PATH to create the temporary directory in.
    """
    return tempfile.mkdtemp(dir=basedir)

def delete_dir(dir):
    """
    Delete a directory

    Parameters
    ----------
    dir : str
          Remove a directory
    """
    shutil.rmtree(dir)
    
def file_search(searchdir,searchstring):
#Recursively search for files in the specified directory
    filelist = []
    for root, dirnames, filenames in os.walk(searchdir):
        for filename in fnmatch.filter(filenames, searchstring):
            filelist.append(os.path.join(root, filename))
    filelist=np.array(filelist)
    return filelist    

# TODO: FIXME
def calculate_slope(x1, x2, y1, y2):
    """

    Parameters
    ----------
    x1
    x2
    y1
    y2

    Returns
    -------

    """
    return (x2-x1)/(y2-y1)

def calculate_slope(x1, x2):
    """

    Parameters
    ----------
    x1
    x2

    Returns
    -------

    """
    slopes = (x2.y.values - x1.y.values)/(x2.x.values-x1.x.values)
    return pd.DataFrame(slopes, columns=['slope'])
