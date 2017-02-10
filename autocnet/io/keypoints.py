import os

import numpy as np
from plio.io import io_hdf

def from_hdf(in_path, node):
    if isinstance(in_path, str):
        hdf = io_hdf.HDFDataset(in_path, mode='r')
    else:
        hdf = in_path

    node.descriptors = hdf['{}/descriptors'.format(node['image_name'])][:]
    raw_kps = hdf['{}/keypoints'.format(node['image_name'])][:]
    index = raw_kps['index']
    clean_kps = utils.remove_field_name(raw_kps, 'index')
    columns = clean_kps.dtype.names

    allkps = pd.DataFrame(data=clean_kps, columns=columns, index=index)

    if 'response' in allkps.columns:
        node._keypoints = allkps.sort_values(by='response', ascending=False)
    elif 'size' in allkps.columns:
        node._keypoints = allkps.sort_values(by='size', ascending=False)
    if isinstance(in_path, str):
        hdf = None

def to_hdf(out_path, node):
    # If the out_path is a string, access the HDF5 file
    if isinstance(out_path, str):
        if os.path.exists(out_path):
            mode = 'a'
        else:
            mode = 'w'
        hdf = io_hdf.HDFDataset(out_path, mode=mode)
    else:
        hdf = out_path

    #try:
    hdf.create_dataset('{}/descriptors'.format(node['image_name']),
                       data=node.descriptors,
                       compression=io_hdf.DEFAULT_COMPRESSION,
                       compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
    hdf.create_dataset('{}/keypoints'.format(node['image_name']),
                       data=hdf.df_to_sarray(node._keypoints.reset_index()),
                       compression=io_hdf.DEFAULT_COMPRESSION,
                       compression_opts=io_hdf.DEFAULT_COMPRESSION_VALUE)
    #except:
        #warnings.warn('Descriptors for the node {} are already stored'.format(self['image_name']))

    # If the out_path is a string, assume this method is being called as a singleton
    # and close the hdf file gracefully.  If an object, let the instantiator of the
    # object close the file
    if isinstance(out_path, str):
        hdf = None

def from_npy(in_path, node):
    nzf = np.load(in_path)
    node.descriptors = nzf['descriptors']
    node._keypoints = pd.DataFrame(nzf['_keypoints'], index=nzf['_keypoints_idx'], columns=nzf['_keypoints_columns'])
    
def to_npy(out_path, node):
    np.savez(out_path, descriptors=node.descriptors,
             _keypoints=data._keypoints,
             _keypoints_idx=data._keypoints.index,
             _keypoints_columns=data._keypoints.columns)
