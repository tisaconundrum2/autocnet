import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../autocnet'))

from autocnet.utils.utils import find_in_dict
from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis, write_filelist
from autocnet.fileio.io_yaml import read_yaml

def read_config(yaml_file):
    config_dict = read_yaml(yaml_file)

    return config_dict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', action='store', help='Provide the name of the file list/adjacency list')
    parser.add_argument('output_file', action='store', help='Provide the name of the output file.')
    args = parser.parse_args()

    return args

def match_images(args, config_dict):

    # Matches the images in the input file using various candidate graph methods
    # produces two files usable in isis
    try:
        cg = CandidateGraph.from_adjacency(find_in_dict(config_dict, 'inputfile_path') +
                                           args.input_file, basepath=find_in_dict(config_dict, 'basepath'))
    except:
        cg = CandidateGraph.from_filelist(find_in_dict(config_dict, 'inputfile_path') + args.input_file)

    # Apply SIFT to extract features
    cg.extract_features(method=find_in_dict(config_dict, 'method'),
                        extractor_parameters={'nfeatures': find_in_dict(config_dict, 'nfeatures')})

    # Match
    cg.match_features()

    # Apply outlier detection
    cg.symmetry_checks()
    cg.ratio_checks()

    # Compute a homography and apply RANSAC
    cg.compute_fundamental_matrices(clean_keys=['ratio', 'symmetry'])

    cg.subpixel_register(clean_keys=['fundamental', 'symmetry', 'ratio'], template_size=5, search_size=15)

    cg.suppress(clean_keys=['fundamental'], k=50)

    cnet = cg.to_cnet(clean_keys=['subpixel'], isis_serials=True)

    filelist = cg.to_filelist()
    write_filelist(filelist, find_in_dict(config_dict, 'outputfile_path') + args.output_file + '.lis')

    to_isis(find_in_dict(config_dict, 'outputfile_path') + args.output_file + '.net', cnet, mode='wb', targetname='Moon')

if __name__ == '__main__':
    config = read_config('/home/acpaquette/autocnet/.image_match_config.yml')
    command_line_args = parse_arguments()
    match_images(command_line_args, config)
