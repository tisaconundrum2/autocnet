import os
import sys

from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist

sys.path.insert(0, os.path.abspath('../autocnet/'))

cg = CandidateGraph.from_adjacency(sys.argv[1], basepath='/home/acpaquette/Desktop/')

# Apply SIFT to extract features
cg.extract_features(method='sift', extractor_parameters={'nfeatures': 500})

# Match
cg.match_features()

# Apply outlier detection
cg.symmetry_checks()
cg.ratio_checks()

m = cg.edge[0][1].masks

# Compute a homography and apply RANSAC
cg.compute_fundamental_matrices(clean_keys=['ratio', 'symmetry'])

cg.subpixel_register(clean_keys=['fundamental', 'symmetry', 'ratio'], template_size=5, search_size=15)

cnet = cg.to_cnet(clean_keys=['subpixel'], isis_serials=True)

filelist = cg.to_filelist()
write_filelist(filelist, 'TestList.lis')

to_isis('TestList.net', cnet, mode='wb', targetname='Moon')
