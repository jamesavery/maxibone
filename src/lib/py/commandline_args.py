#! /usr/bin/python3
'''
This module is used to parse commandline arguments using the `argparse` module.

It is in its own module, to provide a default yet extendable interface for all of the scripts in this project.
'''
import sys
sys.path.append(sys.path[0]+"/../")

import argparse
import config.constants as constants

def default_parser(description='MISSING DESCRIPTION', default_scale=1):
    '''
    This function returns a default `argparse` parser for all scripts in this project.
    The caller can then add more arguments to the parser as needed.
    As a result, the caller must call `parser.parse_args()` to get the arguments.

    Parameters
    ----------
    `description` : str
        A description of the script.
    `default_scale` : int
        The default scale of the image to be processed.

    Returns
    -------
    `parser` : argparse.ArgumentParser
        The default parser.
    '''

    epilog='For more information, please visit github.com/jamesavery/maxibone'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    parser.add_argument('sample', action='store', type=str,
        help='The sample name to be processed, e.g. "770c_pag".')
    parser.add_argument('sample scale', action='store', type=int, default=default_scale, nargs='?',
        help=f'The scale of the image to be processed. Default is {default_scale}.')
    parser.add_argument('-c', '--chunk-length', action='store', type=int, default=256,
        help='The size of the z-axis of the chunks to be processed. Default is 256.')
    parser.add_argument('-v', '--verbose', action='store', type=int, default=0,
        help='Set the verbosity level of the script. Default is 0.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {constants.VERSION}',
        help='Print the version of the script and exit.')

    return parser
