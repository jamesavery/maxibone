#! /usr/bin/python3
'''
This module is used to parse commandline arguments using the `argparse` module.

It is in its own module, to provide a default yet extendable interface for all of the scripts in this project.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent.parent}')

import argparse
import config.constants as constants

def default_parser(description='MISSING DESCRIPTION', default_scale=1):
    '''
    This function returns a default `argparse` parser for all scripts in this project.
    The caller can then add more arguments to the parser as needed.
    As a result, the caller must call `parser.parse_args()` to get the arguments.

    The default arguments are:
    1. `sample` : str
        The sample name to be processed.
    2. `sample_scale` : int
        The scale of the image to be processed. Default is 1.
    3. `chunk_size` : int
        The size of the z-axis of the chunks to be processed. Default is 64.
    4. `verbose` : int
        The verbosity level of the script. Default is 1.
    5. `plotting` : bool
        Whether to plot the results of the script. Default is True.

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
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    parser = add_volume(parser, 'sample', default_scale, None, (None, '?'))
    parser.add_argument('-c', '--chunk-size', action='store', type=int, default=64,
        help='The size of the z-axis of the chunks to be processed. Default is 64.')
    parser.add_argument('--disable-plotting', action='store_false', dest='plotting', default=True,
        help='Disable plotting the results of the script.')
    parser.add_argument('-v', '--verbose', action='store', type=int, default=1,
        help='Set the verbosity level of the script. Default is 1. Generally, 0 is no output, 1 is progress / some text output, 2 is helper/core function output, and 3 is extreme debugging.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {constants.VERSION}',
        help='Print the version of the script and exit.')

    return parser

def add_volume(parser, name, default_scale=1, default_name='', nargs=('?', '?')):
    '''
    This function adds a volume argument to the parser.

    Parameters
    ----------
    `parser` : argparse.ArgumentParser
        The parser to add the volume argument to.
    `name` : str
        The name of the volume to be processed.
    `default_scale` : int
        The default scale of the volume to be processed.
    `default_name` : str
        The default name of the volume to be processed.
    `nargs` : tuple[str,str]
        Whether the arguments should be optional or not.


    Returns
    -------
    `parser` : argparse.ArgumentParser
        The parser with the volume argument added.
    '''

    default_desc = '' if default_name is None or default_name == '' else f'Default is {default_name}.'
    parser.add_argument(f'{name}', action='store', type=str, default=default_name, nargs=nargs[0],
        help=f'The name of the {name} volume to be processed. {default_desc}')
    parser.add_argument(f'{name}_scale', action='store', type=int, default=default_scale, nargs=nargs[1],
        help=f'The scale of the {name} volume to be processed. Default is {default_scale}.')

    return parser