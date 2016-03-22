import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.abspath('../autocnet'))

from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis, write_filelist
import autocnet.fileio.io_yaml


def read_config(yaml_file):
    config_dict = read_yaml(yaml_file)
    return config_dict


# parses command line arguments into a single args variable
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='input_file', default='No_Input', help='Provide the name of the file list/adjacency list')
    parser.add_argument('-o', action='store', dest='output_file', help='Provide the name of the output file')
    parser.add_argument('-p', action='store', dest='basepath', help='Provide the path to the image files')
    args = parser.parse_args()

    return args

def match_images(args):

    # Matches the images in the input file using various candidate graph methods
    # produces two files usable in isis
    try:
        cg = CandidateGraph.from_adjacency(args.input_file, basepath=args.basepath)
    except:
        cg = CandidateGraph.from_filelist(args.input_file)

    # Apply SIFT to extract features
    cg.extract_features(method='sift', extractor_parameters={'nfeatures': 1000})

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
    write_filelist(filelist, args.output_file + '.lis')

    to_isis(args.output_file + '.net', cnet, mode='wb', targetname='Moon')

if __name__ == '__main__':
    config = read_config('/home/acpaquette/test.yml')
    print(config)
    # command_line_args = parse_arguments()
    # match_images(command_line_args)