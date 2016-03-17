import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('../autocnet'))

from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist

if __name__ == '__main__':

    # parses command line arguments into a single args variable
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('input_filename', action='store', help='Provide the name of the file list/adjacency list')
        parser.add_argument('output_filename', action='store', help='Provide the name of the output file')
        args = parser.parse_args()

        return args

    # Matches the images in the input file using various candidate graph methods
    # produces two files usable in isis
    def match_images(args):
        cg = CandidateGraph.from_adjacency(args.input_filename, basepath='/home/acpaquette/Desktop/')

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
        write_filelist(filelist, args.input_filename + '.lis')

        to_isis(args.input_filename + '.net', cnet, mode='wb', targetname='Moon')

        # Ticket calls for a user specified "file list".
        # What kind of "file list" should this ui take? Should it be in the form of a .json file or should we allow the user
        # to enter the images he/she wants to look at in particular and parse them into a .json?

    command_line_args = parse_arguments()
    match_images(command_line_args)