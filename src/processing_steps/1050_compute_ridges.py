#! /usr/bin/python3
'''
This script computes the connected lines in a 2D histogram. It can either be run in GUI mode, where one tries to find the optimal configuration parameters, or batch mode, where it processes one or more histograms into images. In GUI mode, one can specify the bounding box by dragging a box on one of the images, specify the line to highlight by left clicking and reset the bounding box by middle clicking.
'''
# Add the project files to the Python path
import os
import pathlib
import sys
sys.path.append(f'{pathlib.Path(os.path.abspath(__file__)).parent.parent}')
# Ensure that matplotlib does not try to open a window
import matplotlib
matplotlib.use('Agg')

import argparse
import cv2
import json
from lib.py.helpers import row_normalize
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
import time

# TODO At some point, move away from global variables.

def batch():
    '''
    Processes the histograms in batch mode. The histograms are stored in the output folder specified in command line arguments.

    The histograms are stored in the output folder as npz files, where each field is stored as a separate image. The fields are stored as uint8 images, where each pixel is assigned a unique value, corresponding to the connected line it belongs to. The fields are stored as the field name, e.g. "x", "y", "z", "r" or the field name from the histogram file.
    '''

    global args, config

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for histogram in args.histogram:
        sample = ''.join(os.path.basename(histogram).split('.')[:-1])
        f = np.load(histogram)
        tmp = dict()
        axis_names  = ["x","y","z","r"] # TODO: Skriv i compute_histograms
        field_names = f["field_names"]

        for name in axis_names:
            bins = f[f"{name}_bins"]
            rng = _range(0,bins.shape[1],0,bins.shape[0])
            px, py = scatter_peaks(bins, config)
            mask = np.zeros(bins.shape, dtype=np.uint8)
            mask[py, px] = 255
            _, eroded = process_closing(mask, config)
            labeled, _ = process_contours(eroded, rng, config)
            tmp[name] = labeled

        for i,name in enumerate(field_names):
            bins = f["field_bins"][i]
            rng = _range(0,bins.shape[1],0,bins.shape[0])
            px, py = scatter_peaks(bins, config)
            mask = np.zeros(bins.shape, dtype=np.uint8)
            mask[py, px] = 255
            _, eroded = process_closing(mask, config)
            labeled, _ = process_contours(eroded, rng, config)
            tmp[name] = labeled

        np.savez(f'{args.output}/{sample}_labeled', **tmp)

        if args.verbose >= 1:
            print (f'Processed {sample}')

def load_config(filename):
    '''
    Loads the configuration file if it exists, otherwise it returns the default configuration.

    The configuration file is a json file, which stores the following parameters:
    - min peak height: The minimum height of a peak in the line plot to be considered a peak.
    - close kernel y: The y-size of the kernel used for closing the mask.
    - close kernel x: The x-size of the kernel used for closing the mask.
    - line smooth: The size of the gaussian filter used for smoothing the line plot.
    - iter dilate: The number of iterations used for dilating the mask.
    - iter erode: The number of iterations used for eroding the mask.
    - min contour size: The minimum size of a contour to be considered a contour.
    - joint kernel size: The size of the kernel used for finding joints in the skeleton.

    Parameters
    ----------
    `filename` : str
        The filename of the configuration file.

    Returns
    -------
    `config` : dict[str, int]
        The configuration parameters as a dictionary.
    '''

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'min peak height': 1,
            'close kernel y': 10,
            'close kernel x': 2,
            'line smooth': 7,
            'iter dilate': 10,
            'iter erode': 5,
            'min contour size': 1500,
            'joint kernel size': 2
        }

    return config

def load_hists(filename):
    '''
    Loads the histograms from the specified file.

    The histograms are stored in a npz file, where each field is stored as a separate array. The fields are stored as the field name, e.g. "x", "y", "z", "r" or the field name from the histogram file.

    Parameters
    ----------
    `filename` : str
        The filename of the histogram file.

    Returns
    -------
    `hists` : dict[str, numpy.array[uint64]]
        The histograms as a dictionary, where the keys are the field names and the values are the histograms.
    '''

    hists = np.load(filename)
    results = {}
    for name, hist in hists.items():
        hist_sum = np.sum(hist, axis=1)
        hist_sum[hist_sum==0] = 1
        results[name] = hist / hist_sum[:,np.newaxis]

    return hists

class _range:
    '''
    A class for storing the range of the bounding box.

    The class stores the start and stop values for the x and y axis of the bounding box. It also has a method for updating the values from the trackbars in the GUI.

    Attributes
    ----------
    `x` : inner_range
        The range for the x-axis.
    `y` : inner_range
        The range for the y-axis.

    Methods
    -------
    `update()`
        Updates the values of the range from the trackbars in the GUI.
    `__init__(range_x_start=0, range_x_stop=0, range_y_start=0, range_y_stop=0)`
        Initializes the range with the specified values.
    '''

    class inner_range:
        '''
        A class for storing the range of one axis.

        Attributes
        ----------
        `start` : int
            The start value of the range.
        `stop` : int
            The stop value of the range.
        '''

        start = 0
        stop = 0

    x = inner_range()
    y = inner_range()

    def update(self):
        '''
        Updates the values of the range from the trackbars in the GUI.

        Returns
        -------
        `changed` : bool
            Whether the values have changed.
        '''

        changed = False
        tmp = self.x.start
        self.x.start = cv2.getTrackbarPos('range start x', 'Histogram lines')
        changed |= tmp == self.x.start
        tmp = self.x.stop
        self.x.stop  = cv2.getTrackbarPos('range stop x', 'Histogram lines')
        changed |= tmp == self.x.stop
        tmp = self.y.start
        self.y.start = cv2.getTrackbarPos('range start y', 'Histogram lines')
        changed |= tmp == self.y.start
        tmp = self.y.stop
        self.y.stop  = cv2.getTrackbarPos('range stop y', 'Histogram lines')
        changed |= tmp == self.y.stop

        return changed, self

    def __init__(self, range_x_start=0, range_x_stop=0, range_y_start=0, range_y_stop=0):
        '''
        Initializes the range with the specified values.

        Parameters
        ----------
        `range_x_start` : int
            The start value of the x-axis.
        `range_x_stop` : int
            The stop value of the x-axis.
        `range_y_start` : int
            The start value of the y-axis.
        `range_y_stop` : int
            The stop value of the y-axis.
        '''

        self.x.start = range_x_start
        self.x.stop  = range_x_stop
        self.y.start = range_y_start
        self.y.stop  = range_y_stop

def parse_args():
    '''
    Parses the command line arguments.

    Returns
    -------
    `args` : argparse.Namespace
        The parsed command line arguments.
    '''

    parser = argparse.ArgumentParser(description="""Computes the connected lines in a 2D histogram. It can either be run in GUI mode, where one tries to find the optimal configuration parameters, or batch mode, where it processes one or more histograms into images. In GUI mode, one can specify the bounding box by dragging a box on one of the images, specify the line to highlight by left clicking and reset the bounding box by middle clicking.

Example command for running with default configuration:
python src/histogram_processing/compute_ridges.py $BONE_DATA/processed/histograms/770c_pag/bins-bone_region3.npz -b -o $BONE_DATA/processed/histograms/770c_pag/
""")

    # E.g. histogram: /mnt/data/MAXIBONE/Goats/tomograms/processed/histograms/770c_pag/bins1.npz
    # TODO glob support / -r --recursive
    # TODO change to match the other argparse's
    parser.add_argument('histogram', nargs='+',
        help='Specifies one or more histogram files (usually *.npz) to process. If in GUI mode, only the first will be processed. Glob is currently not supported.')
    parser.add_argument('-b', '--batch', action='store_true',
        help='Toggles whether the script should be run in batch mode. In this mode, the GUI isn\'t launched, but the provided histograms will be stored in the specified output folder.')
    parser.add_argument('-c', '--config', default='config_compute_ridges.json', type=str,
        help='The configuration file storing the parameters. If in GUI mode, this will be overwritten with the values on the trackbars (unless the -b flag is provided). If it doesn\'t exist, the default values inside this script will be used. NOTE: there\'s an error in libxkbcommon.so, which crashes OpenCV whenever any other key than escape is used. So to save it, close the GUI with the escape button.')
    parser.add_argument('-d', '--dry_run', action='store_true',
        help='Toggles whether the configuration should be saved, when running in GUI mode.')
    parser.add_argument('--disable-matplotlib', action='store_true',
        help='Toggles whether the matplotlib plots should be shown')
    parser.add_argument('-o', '--output', default='output', type=str,
        help='Specifies the folder to put the resulting images in.')
    parser.add_argument('-p', '--peaks', action='store_true',
        help="Toggles whether to plot peaks on the cutout (1st row, 3rd image). Turned off by default, since it is computationally heavy. Only applicable in GUI mode.")
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Toggles whether debug printing should be enabled.')

    args = parser.parse_args()

    return args

def plot_line(line, rng: _range):
    '''
    Plots the line and the meaned line with the peaks highlighted.

    The line is plotted as a line plot, where the meaned line is plotted on top of it. The peaks are highlighted with red dots.

    Parameters
    ----------
    `line` : numpy.array[uint64]
        The line to plot.
    `rng` : _range
        The range of the bounding box.

    Returns
    -------
    `line_plot` : numpy.array[uint8]
        The line plot as an image.
    '''

    global config, disable_matplotlib

    if disable_matplotlib:
        line_plot = np.zeros((3,3,3), dtype=np.uint8)
    else:
        meaned, peaks = process_line(line[rng.x.start:rng.x.stop], config)
        fig, ax = plt.subplots()
        ax.plot(line[rng.x.start:rng.x.stop], zorder=0)
        ax.plot(meaned, zorder=1)
        ax.scatter(peaks, meaned[peaks], c='red', zorder=2)
        line_plot = mplfig_to_npimage(fig)
        plt.close()
        line_plot = cv2.cvtColor(line_plot, cv2.COLOR_RGB2BGR)

    return line_plot

def process_closing(mask, config):
    '''
    Processes the mask by closing (dilate followed by erode).

    The mask is first dilated and then eroded with a cross kernel of the specified size. The number of iterations for dilating and eroding can also be specified in the configuration.

    Parameters
    ----------
    `mask` : numpy.array[uint8]
        The mask to process.
    `config` : dict[str, int]
        The configuration parameters.

    Returns
    -------
    `dilated` : numpy.array[uint8]
        The dilated mask.
    `eroded` : numpy.array[uint8]
        The eroded mask (after dilation).
    '''

    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (config['close kernel x'], config['close kernel y']))
    dilated = cv2.dilate(mask, close_kernel, iterations=config['iter dilate'])
    eroded = cv2.erode(dilated, close_kernel, iterations=config['iter erode'])

    return dilated, eroded

def process_contours(hist, rng: _range, config):
    '''
    Processes the histogram by finding the contours.

    The histogram is first thresholded to find the contours. The contours are then filtered by size, where the minimum size can be specified in the configuration. The contours are then sorted by x-coordinate and drawn on a blank image.

    Parameters
    ----------
    `hist` : numpy.array[uint64]
        The histogram to process.
    `rng` : _range
        The range of the bounding box.
    `config` : dict[str, int]
        The configuration parameters.

    Returns
    -------
    `result` : numpy.array[uint8]
        The image with the contours drawn.
    '''

    contours, _ = cv2.findContours(hist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sizes = [cv2.arcLength(c, True) for c in contours]
    contours_filtered = [contours[i] for i, size in enumerate(contours_sizes) if size > config['min contour size']]
    result = np.zeros((rng.y.stop-rng.y.start, rng.x.stop-rng.x.start), np.uint8)
    bounding_boxes = [cv2.boundingRect(c) for c in contours_filtered]
    if args.verbose >= 2: print (len(contours_filtered), len(bounding_boxes))
    (contours_filtered, _) = zip(*sorted(zip(contours_filtered, bounding_boxes), key=lambda b:b[1][0]))
    for i in np.arange(len(contours_filtered)):
        result = cv2.drawContours(result, contours_filtered, i, int(i+1), -1)

    return result, [i+1 for i in range(len(contours_filtered))]

def process_joints(hist):
    '''
    Processes the histogram by finding the joints in the skeleton.

    The histogram is first thresholded to find the skeleton. The skeleton is then dilated with a horizontal and vertical kernel. The joints are then found by taking the intersection of the horizontal and vertical dilated skeletons.

    Parameters
    ----------
    `hist` : numpy.array[uint64]
        The histogram to process.

    Returns
    -------
    `joints` : numpy.array[uint8]
        The image with the joints drawn.
    '''

    global config

    tmp = cv2.threshold(hist, 1, 1, cv2.THRESH_BINARY)[1]
    skeleton = skeletonize(tmp)
    skeleton = skeleton.astype(np.uint8) * 255
    skeleton_gray = cv2.dilate(skeleton, (3,3), 100)
    joint_kernel_size = config['joint kernel size']
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (joint_kernel_size,1))
    horizontal = cv2.morphologyEx(skeleton_gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,joint_kernel_size))
    vertical = cv2.morphologyEx(skeleton_gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    joints = cv2.bitwise_and(horizontal, vertical)
    joints = cv2.dilate(joints, (10,10), iterations=5)

    return joints

def process_line(line, config):
    '''
    Processes the line by smoothing it and finding the peaks.

    The line is first smoothed with a gaussian filter. The peaks are then found by the scipy function `find_peaks`.

    Parameters
    ----------
    `line` : numpy.array[uint64]
        The line to process.
    `config` : dict[str, int]
        The configuration parameters.

    Returns
    -------
    `meaned` : numpy.array[float64]
        The smoothed line.
    `peaks` : list[int]
        The indices of the peaks.
    '''

    meaned = gaussian_filter1d(line, config['line smooth'])
    peaks, _ = signal.find_peaks(meaned, .01*config['min peak height']*line.max())

    return meaned, peaks

def process_scatter_peaks(hist, rng: _range, peaks=None):
    '''
    Processes the histogram by finding the peaks and plotting them.

    The histogram is first smoothed with a gaussian filter. The peaks are then found by the scipy function `find_peaks`. The peaks are then scattered on the cutout of the original histogram.

    Parameters
    ----------
    `hist` : numpy.array[uint64]
        The histogram to process.
    `rng` : _range
        The range of the bounding box.
    `peaks` : list[int]
        The peaks to scatter. If None, the peaks are found.

    Returns
    -------
    `scatter_plot` : numpy.array[uint8]
        The image with the peaks scattered.
    `px` : list[int]
        The x-coordinates of the peaks.
    `py` : list[int]
        The y-coordinates of the peaks.
    '''

    global config, disable_matplotlib

    # Show the peaks scattered on the cutout of the original
    px, py = scatter_peaks(hist[rng.y.start:rng.y.stop,rng.x.start:rng.x.stop], config)
    if disable_matplotlib:
        scatter_plot = np.zeros((3,3,3), dtype=np.uint8)
    else:
        fig, ax = plt.subplots()
        ax.imshow(row_normalize(hist[rng.y.start:rng.y.stop,rng.x.start:rng.x.stop]), cmap='jet')
        if peaks:
            ax.scatter(px, py, color='red', alpha=.008)
        scatter_plot = mplfig_to_npimage(fig)
        plt.close()
        scatter_plot = cv2.cvtColor(scatter_plot, cv2.COLOR_RGB2BGR)

    return scatter_plot, px, py

def process_with_box(hist, rng: _range, selected_line, scale_x, scale_y, partial_size):
    '''
    Processes the histogram by plotting the histogram with the bounding box and the selected line.

    The histogram is first normalized and scaled to 255. The bounding box is then drawn on the histogram. The selected line is also drawn on the histogram.

    Parameters
    ----------
    `hist` : numpy.array[uint64]
        The histogram to process.
    `rng` : _range
        The range of the bounding box.
    `selected_line` : int
        The selected line to highlight.
    `scale_x` : float
        The scaling factor for the x-axis.
    `scale_y` : float
        The scaling factor for the y-axis.
    `partial_size` : tuple[int]
        The size of the partial image.

    Returns
    -------
    `colored` : numpy.array[uint8]
        The image with the histogram, bounding box and selected line drawn.
    '''

    display = (row_normalize(hist) * 255).astype(np.uint8)
    box_x_start = int(rng.x.start * scale_x)
    box_y_start = int(rng.y.start * scale_y)
    box_x_stop = int(rng.x.stop * scale_x)
    box_y_stop = int(rng.y.stop * scale_y)
    if args.verbose >= 2: print (partial_size)
    resized = cv2.resize(display, partial_size)
    colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    colored = cv2.rectangle(colored, (box_x_start, box_y_start), (box_x_stop, box_y_stop), (0, 255, 0), 1)
    colored[int(selected_line*scale_y),:] = (0,0,255)

    return colored

def save_config():
    '''
    Saves the configuration to the configuration file.

    The configuration is saved as a json file at the location specified in the command line arguments.
    '''

    global config, args

    with open(args.config, 'w') as f:
        json.dump(config, f)

def scatter_peaks(hist, config):
    '''
    Scatters the peaks on the histogram.

    The histogram is first smoothed with a gaussian filter. The peaks are then found by the scipy function `find_peaks`. The peaks are then scattered on the histogram.

    Parameters
    ----------
    `hist` : numpy.array[uint64]
        The histogram to process.
    `config` : dict[str, int]
        The configuration parameters.

    Returns
    -------
    `px` : list[int]
        The x-coordinates of the peaks.
    `py` : list[int]
        The y-coordinates of the peaks.
    '''

    meaned = gaussian_filter1d(hist, config['line smooth'], axis=1)
    maxs = meaned.max(axis=1)*.01*config['min peak height']
    fp = lambda x,y: signal.find_peaks(x, y)[0]
    peaks_x = [fp(row, maxval) for row, maxval in zip(meaned, maxs)]
    peaks_y = [[i]*len(peaks) for i, peaks in enumerate(peaks_x)]
    flatten = lambda x: [elem for li in x for elem in li]

    return flatten(peaks_x), flatten(peaks_y)

def gui():
    '''
    Runs the GUI for finding the optimal configuration parameters.

    The GUI consists of several trackbars, where the user can adjust the parameters. The user can also drag a box on the histogram to specify the bounding box, left click to specify the line to highlight and middle click to reset the bounding box. The user can also save the configuration by pressing the 's' key.

    The GUI is updated whenever the trackbars are changed.

    The GUI is closed by pressing the escape key, which also saves the configuration if the 'dry run' commandline argument is not set.
    '''

    global last, args, ready

    ready = False

    def update_image(_):
        hist_shape = hists[cv2.getTrackbarPos('bins', 'Histogram lines')].shape
        cv2.setTrackbarMax('range start x', 'Histogram lines', hist_shape[1]-1)
        cv2.setTrackbarPos('range start x', 'Histogram lines', min(cv2.getTrackbarPos('range start x', 'Histogram lines'), hist_shape[1]-1))
        cv2.setTrackbarMax('range stop x', 'Histogram lines', hist_shape[1]-1)
        cv2.setTrackbarPos('range stop x', 'Histogram lines', min(cv2.getTrackbarPos('range stop x', 'Histogram lines'), hist_shape[1]-1))
        cv2.setTrackbarMax('range start y', 'Histogram lines', hist_shape[0]-1)
        cv2.setTrackbarPos('range start y', 'Histogram lines', min(cv2.getTrackbarPos('range start y', 'Histogram lines'), hist_shape[0]-1))
        cv2.setTrackbarMax('range stop y', 'Histogram lines', hist_shape[0]-1)
        cv2.setTrackbarPos('range stop y', 'Histogram lines', min(cv2.getTrackbarPos('range stop y', 'Histogram lines'), hist_shape[0]-1))
        update(True)

    def update_line(event, x, y, flags, param):
        '''
        Updates the line and bounding box when the user interacts with the GUI.

        The line is updated when the user left clicks on the histogram. The bounding box is updated when the user drags a box on the histogram. The bounding box is reset when the user middle clicks on the histogram.

        Parameters
        ----------
        `event` : int
            The event type.
        `x` : int
            The x-coordinate of the event.
        `y` : int
            The y-coordinate of the event.
        `flags` : int
            The flags of the event, passed by OpenCV.
        `param` : int
            The parameter of the event, passed by OpenCV.

        Returns
        -------
        `None`
        '''

        global mx, my, scale_x, scale_y, disable_matplotlib

        width_factor = 3
        height_factor = 1 if disable_matplotlib else 2
        sizex = cv2.getTrackbarPos('size x', 'Histogram lines')
        sizey = cv2.getTrackbarPos('size y', 'Histogram lines')
        _, rng = _range().update()
        rwidth = rng.x.stop - rng.x.start
        rheight = rng.y.stop - rng.y.start
        pwidth = sizex // width_factor
        pheight = sizey // height_factor
        dead_zone = 100 # This is in global scale
        if y > sizey // height_factor or x < sizex // width_factor:
            lx = x % pwidth
            ly = y % pheight
            gx = int(lx * (1./scale_x)) if y < sizey // height_factor else int(rng.x.start + lx * (1./(pwidth / rwidth)))
            gy = int(ly * (1./scale_y)) if y < sizey // height_factor else int(rng.y.start + ly * (1./(pheight / rheight)))
            if (event == cv2.EVENT_LBUTTONDOWN):
                if args.verbose >= 2: print ('down', gx, gy)
                mx = gx
                my = gy
            elif (event == cv2.EVENT_LBUTTONUP):
                if args.verbose >= 2: print ('up', gx, gy)
                if abs(mx-gx) < dead_zone and abs(my-gy) < dead_zone:
                    cv2.setTrackbarPos('line', 'Histogram lines', gy)
                    if args.verbose >= 2: print ('Set line', y, ly, gy)
                    #update(42)
                else:
                    mx, gx = sorted((mx,gx))
                    my, gy = sorted((my,gy))
                    cv2.setTrackbarPos('range start x', 'Histogram lines', mx)
                    cv2.setTrackbarPos('range stop x', 'Histogram lines', gx)
                    cv2.setTrackbarPos('range start y', 'Histogram lines', my)
                    cv2.setTrackbarPos('range stop y', 'Histogram lines', gy)
                    if args.verbose >= 2: print ('Set bounds', mx, my, gy, gy)
                    update(force=True)
            elif (event == cv2.EVENT_MBUTTONDOWN):
                if args.verbose >= 2: print ('reset bounding box')
                hist_shape = hists[cv2.getTrackbarPos('bins', 'Histogram lines')].shape
                cv2.setTrackbarPos('range start x', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop x', 'Histogram lines', hist_shape[1]-1)
                cv2.setTrackbarPos('range start y', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop y', 'Histogram lines', hist_shape[0]-1)
                #update(42)

    # TODO Only do recomputation, whenever parameters that have an effect are changed.
    def update(force=False): # Note: all colors are in BGR format, as this is what OpenCV uses
        '''
        Updates the GUI.

        The GUI is updated by processing the histogram with the specified parameters. The histogram is then displayed in the GUI.

        Parameters
        ----------
        `force` : bool
            Whether the GUI should be updated even if the parameters haven't changed.

        Returns
        -------
        `None`
        '''

        global last, config, scale_x, scale_y, disable_matplotlib, ready

        times = []
        modified = update_config() or force
        if args.verbose >= 2: print ('----- ready', ready, modified, force)
        if not ready or not modified:
            return
        times.append(('init',time.time()))

        # Check if trackbar ranges should be updated
        bin = cv2.getTrackbarPos('bins', 'Histogram lines')
        hist = hists[bin]
        if bin != last['bin']:
            modified = True
            last['bin'] = bin
            # Set trackbar max values
            cv2.setTrackbarMax('range start x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarMax('range stop x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarMax('range start y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarMax('range stop y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarMax('line', 'Histogram lines', hist.shape[0]-1)

            # Set trackbar starting values
            cv2.setTrackbarPos('range start x', 'Histogram lines', 0)
            cv2.setTrackbarPos('range stop x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarPos('range start y', 'Histogram lines', 0)
            cv2.setTrackbarPos('range stop y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarPos('line', 'Histogram lines', 0)

        # Show the original image, along with the line
        updated_rng, rng = _range().update()
        selected_line = cv2.getTrackbarPos('line', 'Histogram lines')
        modified |= updated_rng

        times.append(('trackbar parsing',time.time()))

        partial_width = cv2.getTrackbarPos('size x', 'Histogram lines') // 3
        if disable_matplotlib:
            partial_height = cv2.getTrackbarPos('size y', 'Histogram lines')
        else:
            partial_height = cv2.getTrackbarPos('size y', 'Histogram lines') // 2
        if partial_width != last['partial_width'] or partial_height != last['partial_height'] or selected_line != last['selected_line'] or modified:
            last['partial_width'] = partial_width
            last['partial_height'] = partial_height
            last['selected_line'] = selected_line
            modified = True
            partial_size = (partial_width, partial_height)
            scale_x = partial_width / hist.shape[1]
            scale_y = partial_height / hist.shape[0]

            colored = process_with_box(hist, rng, selected_line, scale_x, scale_y, partial_size)
            last['colored'] = colored
        else:
            colored = last['colored']
            partial_size = (last['partial_width'], last['partial_height'])

        times.append(('processed with box',time.time()))

        # Show the plot of a single line
        line_params = ['line smooth', 'min peak height']
        line_changed = [last[param] != config[param] for param in line_params]
        if any(line_changed) or modified:
            #modified = True # TODO this should not affect the later ones...? It should actually.
            line_plot = plot_line(hist[selected_line, :], rng)
            lp_resized = cv2.resize(line_plot, partial_size)
            last['lp_resized'] = lp_resized
            for param in line_params:
                last[param] = config[param]
        else:
            lp_resized = last['lp_resized']

        times.append(('plot line',time.time()))

        if modified:
            modified = True
            scatter_plot, px, py = process_scatter_peaks(hist, rng, peaks=args.peaks)
            sp_resized = cv2.resize(scatter_plot, partial_size)
            last['sp_resized'] = sp_resized
            last['px'] = px
            last['py'] = py
        else:
            sp_resized = last['sp_resized']
            px = last['px']
            py = last['py']

        times.append(('scatter peaks',time.time()))

        if not disable_matplotlib:
            first_row = np.concatenate((colored, lp_resized, sp_resized), axis=1)

        close_params = ['close kernel x', 'close kernel y', 'iter erode', 'iter dilate']
        changed = [config[entry] != last[entry] for entry in close_params]

        local_scale_y = partial_height / (rng.y.stop - rng.y.start)
        if any(changed) or modified:
            for param in close_params:
                last[param] = config[param]
            modified = True
            mask = np.zeros((rng.y.stop-rng.y.start, rng.x.stop-rng.x.start), dtype=np.uint8)
            mask[py, px] = 255

            _, eroded = process_closing(mask, config)
            display_eroded = cv2.resize(eroded, partial_size)
            display_eroded = cv2.cvtColor(display_eroded, cv2.COLOR_GRAY2BGR)
            if selected_line > rng.y.start and selected_line < rng.y.stop:
                display_eroded[int((selected_line-rng.y.start)*local_scale_y),:] = (0,0,255)
            last['eroded'] = eroded
            last['display_eroded'] = display_eroded
        else:
            eroded = last['eroded']
            display_eroded = last['display_eroded']

        times.append(('closing',time.time()))

        # Find lines
        if config['min contour size'] != last['min contour size'] or modified:
            last['min contour size'] = config['min contour size']
            modified = True
            labeled, labels = process_contours(eroded, rng, config)
            label_colours = [(0,0,0)] + [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in labels]
            labeled_colour = np.zeros((labeled.shape[0], labeled.shape[1], 3), dtype=np.uint8)
            for l in labels:
                labeled_colour[labeled == l] = label_colours[l]
            display_contours = cv2.resize(labeled_colour, partial_size)
            if selected_line > rng.y.start and selected_line < rng.y.stop:
                display_contours[int((selected_line-rng.y.start)*local_scale_y),:] = (0,0,255)
            last['display_contours'] = display_contours
        else:
            display_contours = last['display_contours']

        times.append(('find lines',time.time()))

        if disable_matplotlib:
            cv2.imshow('Histogram lines', np.concatenate((colored, display_eroded, display_contours), axis=1))
        else:
            second_row = np.concatenate((display_eroded, display_contours, np.zeros_like(display_contours)), axis=1)
            cv2.imshow('Histogram lines', np.concatenate((first_row, second_row)))

        for i in range(1, len(times)):
            label, tim = times[i]
            if args.verbose >= 2: print (label, tim - times[i-1][1])

    def update_config():
        '''
        Updates the configuration from the trackbars.

        Returns
        -------
        `changed` : bool
            Whether the configuration has changed.
        '''

        global config, ready

        if not ready:
            return False

        changed : bool = False
        for entry in config.keys():
            tmp = config[entry]
            config[entry] = cv2.getTrackbarPos(entry, 'Histogram lines')
            if args.verbose >= 2: print (entry, tmp, config[entry])
            changed = changed or (tmp != config[entry])

        if changed and args.verbose >= 2: print ('Changed!');

        return changed

    f = np.load(args.histogram[0])
    keys = [key for key in f.keys() if key.endswith('_bins') and not key == 'field_bins']
    hists = [f[key] for key in keys]
    hists += list(f['field_bins'])

    first_hist = hists[0]
    last = {
        # Trackbars
        'bin': None,
        'partial_width': None,
        'partial_height': None,
        'selected_line': None,
        'close kernel x': None,
        'close kernel y': None,
        'iter dilate': None,
        'iter erode': None,
        'min contour size': None,
        'line smooth': None,
        'min peak height': None,

        # First row
        'px': None,
        'py': None,
        'colored': None,
        'lp_resized': None,
        'sp_resized': None,

        # Second row
        'eroded': None,
        'display_eroded': None,
        'display_contours': None
    }
    cv2.namedWindow('Histogram lines')
    cv2.createTrackbar('range start x', 'Histogram lines', 0, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range stop x', 'Histogram lines', first_hist.shape[1]-1, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range start y', 'Histogram lines', 0, first_hist.shape[0]-1, update)
    cv2.createTrackbar('range stop y', 'Histogram lines', first_hist.shape[0]-1, first_hist.shape[0]-1, update)
    cv2.createTrackbar('size x', 'Histogram lines', 1280, 1920, update)
    cv2.createTrackbar('size y', 'Histogram lines', 720, 1080, update)
    cv2.createTrackbar('bins', 'Histogram lines', 0, len(hists)-1, update_image)
    cv2.createTrackbar('line', 'Histogram lines', 0, first_hist.shape[0]-1, update)
    cv2.createTrackbar('line smooth', 'Histogram lines', config['line smooth'], 50, update)
    cv2.createTrackbar('min peak height', 'Histogram lines', config['min peak height'], 100, update)
    cv2.createTrackbar('close kernel x', 'Histogram lines', config['close kernel x'], 100, update)
    cv2.createTrackbar('close kernel y', 'Histogram lines', config['close kernel y'], 100, update)
    cv2.createTrackbar('iter dilate', 'Histogram lines', config['iter dilate'], 100, update)
    cv2.createTrackbar('iter erode', 'Histogram lines', config['iter erode'], 100, update)
    cv2.createTrackbar('min contour size', 'Histogram lines', config['min contour size'], 10000, update)
    cv2.createTrackbar('joint kernel size', 'Histogram lines', config['joint kernel size'], 100, update)
    cv2.setMouseCallback('Histogram lines', update_line)
    ready = True
    update(True)

    while True:
        key = cv2.waitKey(16)
        key &= 0xFF
        if key == 113:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    disable_matplotlib = args.disable_matplotlib

    if args.batch:
        batch()
    else:
        # Preload pyplot, as the first figure takes around 3 seconds to compute. For timing only
        plt.figure()
        plt.close()
        gui()
        if not args.dry_run:
            save_config()
