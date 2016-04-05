import argparse

from autocnet.graph.network import CandidateGraph
from autocnet.fileio.io_controlnetwork import to_isis, write_filelist
from autocnet.fileio.io_yaml import read_yaml

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
        cg = CandidateGraph.from_adjacency(config_dict['inputfile_path'] +
                                           args.input_file, basepath=config['basepath'])
    except:
        cg = CandidateGraph.from_filelist(config_dict['inputfile_path'] + args.input_file)

    # Apply SIFT to extract features
    cg.extract_features(method='sift', extractor_parameters={'nfeatures': 1000})

    # Match
    cg.match_features()

    # Apply outlier detection
    cg.apply_func_to_edges('symmetry_check')
    cg.apply_func_to_edges('ratio_check')

    # Compute a homography and apply RANSAC
    cg.apply_func_to_edges('compute_fundamental_matrix', clean_keys=['ratio', 'symmetry'])

    cg.apply_func_to_edges('subpixel_register', clean_keys=['fundamental', 'symmetry', 'ratio'], template_size=5, search_size=15)

    cg.apply_func_to_edges('suppress', clean_keys=['fundamental'], k=50)

    cnet = cg.to_cnet(clean_keys=['subpixel'], isis_serials=True)

    filelist = cg.to_filelist()
    write_filelist(filelist, config_dict['outputfile_path'] + args.output_file + '.lis')

    to_isis(config_dict['outputfile_path'] + args.output_file + '.net', cnet, mode='wb', targetname='Moon')

if __name__ == '__main__':
    config = read_yaml('image_match_config.yml')
    command_line_args = parse_arguments()
    match_images(command_line_args, config)
