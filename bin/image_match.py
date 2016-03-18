import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../autocnet'))

from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist

# parses command line arguments into a single args variable
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='input_file', default='No_Input', help='Provide the name of the file list/adjacency list')
    parser.add_argument('-o', action='store', dest='output_file', help='Provide the name of the output file')
    args = parser.parse_args()
    print(args)

    return args


def match_images(args):

    # Matches the images in the input file using various candidate graph methods
    # produces two files usable in isis
    try:
        cg = CandidateGraph.from_adjacency(args.input_file, basepath='/home/acpaquette/Desktop/')
    except:
        print('here')
        cg = CandidateGraph.from_filelist(args.input_file)

    # Apply SIFT to extract features
    cg.extract_features(method='sift', extractor_parameters={'nfeatures': 1000})

    # Match
    cg.match_features()

    # Apply outlier detection
    cg.symmetry_checks()
    cg.ratio_checks()

    m = cg.edge[0][1].masks

    # Compute a homography and apply RANSAC
    cg.compute_fundamental_matrices(clean_keys=['ratio', 'symmetry'])

    cg.subpixel_register(clean_keys=['fundamental', 'symmetry', 'ratio'], template_size=5, search_size=15)

    cg.suppress(clean_keys=['fundamental'], k=50)

    cnet = cg.to_cnet(clean_keys=['subpixel'], isis_serials=True)

    filelist = cg.to_filelist()
    write_filelist(filelist, args.output_file + '.lis')

    to_isis(args.output_file + '.net', cnet, mode='wb', targetname='Moon')

if __name__ == '__main__':
    command_line_args = parse_arguments()
    match_images(command_line_args)

'''
    file = CandidateGraph.from_filelist(["/home/acpaquette/Desktop/AS15-M-0414_sub4.cub",
      "/home/acpaquette/Desktop/AS15-M-0413_sub4.cub",
      "/home/acpaquette/Desktop/AS15-M-0412_sub4.cub"])

    file2 = CandidateGraph.from_filelist('/home/acpaquette/autocnet/autocnet/examples/Apollo15/user_image_list.json')
'''