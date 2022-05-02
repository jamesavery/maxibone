import cv2
import numpy as np
from numpy.lib.function_base import disp
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import json
import argparse
import os
import time

def batch():
    global args, config

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for histogram in args.histogram:
        sample = ''.join(os.path.basename(histogram).split('.')[:-1])
        f = np.load(histogram)
        tmp = dict()
        for name, bins in f.items():
            rng = _range(0,bins.shape[1],0,bins.shape[0])
            px, py = scatter_peaks(bins, config)
            mask = np.zeros(bins.shape, dtype=np.uint8)
            mask[py, px] = 255
            dilated, eroded = process_closing(mask, config)
            labeled, _ = process_contours(eroded, rng, config)
            tmp[name] = labeled

        np.savez(f'{args.output}/{sample}_labeled', **tmp)
        #np.save(f'{args.output}/{sample}_{name}_labeled', labeled)

        if args.verbose:
            print (f'Processed {sample}')

def load_config(filename):
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
            'min contour size': 2000,
            'joint kernel size': 2
        }
    return config

def load_hists(filename):
    hists = np.load(filename)
    results = {}
    for name, hist in hists.items():
        hist_sum = np.sum(hist, axis=1)
        hist_sum[hist_sum==0] = 1
        results[name] = hist / hist_sum[:,np.newaxis]
    return hists

class _range:
    class inner_range:
        start = 0
        stop = 0

    x = inner_range()
    y = inner_range()

    def update(self):
        self.x.start = cv2.getTrackbarPos('range start x', 'Histogram lines')
        self.x.stop  = cv2.getTrackbarPos('range stop x', 'Histogram lines')
        self.y.start = cv2.getTrackbarPos('range start y', 'Histogram lines')
        self.y.stop  = cv2.getTrackbarPos('range stop y', 'Histogram lines')
        return self

    def __init__(self, range_x_start=0, range_x_stop=0, range_y_start=0, range_y_stop=0):
        self.x.start = range_x_start
        self.x.stop  = range_x_stop
        self.y.start = range_y_start
        self.y.stop  = range_y_stop

def parse_args():
    parser = argparse.ArgumentParser(description="Computes the connected lines in a 2D histogram. It can either be run in GUI mode, where one tries to find the optimal configuration parameters, or batch mode, where it processes one or more histograms into images. In GUI mode, one can specify the bounding box by dragging a box on one of the images, specify the line to highlight by left clicking and reset the bounding box by middle clicking.")

    # E.g. histogram: /mnt/data/MAXIBONE/Goats/tomograms/processed/histograms/770c_pag/bins1.npz
    # TODO glob support / -r --recursive
    parser.add_argument('histogram', nargs='+',
        help='Specifies one or more histogram files (usually *.npz) to process. If in GUI mode, only the first will be processed. Glob is currently not supported.')
    parser.add_argument('-b', '--batch', action='store_true',
        help='Toggles whether the script should be run in batch mode. In this mode, the GUI isn\'t launched, but the provided histograms will be stored in the specified output folder.')
    parser.add_argument('-c', '--config', default='config.json', type=str,
        help='The configuration file storing the parameters. If in GUI mode, this will be overwritten with the values on the trackbars (unless the -b flag is provided). If it doesn\'t exist, the default values inside this script will be used. NOTE: there\'s an error in libxkbcommon.so, which crashes OpenCV whenever any other key than escape is used. So to save it, close the GUI with the escape button.')
    parser.add_argument('-d', '--dry_run', action='store_true',
        help='Toggles whether the configuration should be saved, when running in GUI mode.')
    parser.add_argument('-o', '--output', default='output', type=str,
        help='Specifies the folder to put the resulting images in.')
    parser.add_argument('-p', '--peaks', action='store_true',
        help="Toggles whether to plot peaks on the cutout (1st row, 3rd image). Turned off by default, since it is computationally heavy. Only applicable in GUI mode.")
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Toggles whether debug printing should be enabled.')

    args = parser.parse_args()
    return args

def plot_line(line, rng: _range):
    global config

    meaned, peaks = process_line(line[rng.x.start:rng.x.stop], config)
    fig, ax = plt.subplots()
    ax.plot(line[rng.x.start:rng.x.stop])
    ax.plot(meaned)
    ax.scatter(peaks, meaned[peaks], c='red')
    line_plot = mplfig_to_npimage(fig)
    line_plot = cv2.cvtColor(line_plot, cv2.COLOR_RGB2BGR)
    plt.close()

    return line_plot

def process_closing(mask, config):
    #close_kernel = np.ones((config['close kernel y'], config['close kernel x']))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (config['close kernel x'], config['close kernel y']))
    dilated = cv2.dilate(mask, close_kernel, iterations=config['iter dilate'])
    eroded = cv2.erode(dilated, close_kernel, iterations=config['iter erode'])

    return dilated, eroded

def process_contours(hist, rng: _range, config):
    contours, _ = cv2.findContours(hist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.arcLength(c, True) > config['min contour size']]
    result = np.zeros((rng.y.stop-rng.y.start, rng.x.stop-rng.x.start), np.uint8)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, _) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b:b[1][0]))
    for i in np.arange(len(contours)):
        result = cv2.drawContours(result, contours, i, int(i+1), -1)

    return result, [i+1 for i in range(len(contours))]

def process_joints(hist):
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
    meaned = gaussian_filter1d(line, config['line smooth'])
    peaks, _ = signal.find_peaks(meaned, .01*config['min peak height']*line.max())

    return meaned, peaks

def process_scatter_peaks(hist, rng: _range, peaks=None):
    global config

    # Show the peaks scattered on the cutout of the original
    px, py = scatter_peaks(hist[rng.y.start:rng.y.stop,rng.x.start:rng.x.stop], config)
    fig, ax = plt.subplots()
    ax.imshow(hist[rng.y.start:rng.y.stop,rng.x.start:rng.x.stop], cmap='jet')
    if peaks:
        ax.scatter(px, py, color='red', alpha=.008)
    scatter_plot = mplfig_to_npimage(fig)
    scatter_plot = cv2.cvtColor(scatter_plot, cv2.COLOR_RGB2BGR)
    plt.close()

    return scatter_plot, py, px

def process_with_box(hist, rng: _range, selected_line, scale_x, scale_y, partial_size):
    display = ((hist.astype(np.float32) / hist.max()) * 255.0).astype(np.uint8)
    box_x_start = int(rng.x.start * scale_x)
    box_y_start = int(rng.y.start * scale_y)
    box_x_stop = int(rng.x.stop * scale_x)
    box_y_stop = int(rng.y.stop * scale_y)
    resized = cv2.resize(display, partial_size)
    colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    colored = cv2.rectangle(colored, (box_x_start, box_y_start), (box_x_stop, box_y_stop), (0, 255, 0), 1)
    colored[int(selected_line*scale_y),:] = (0,0,255)

    return colored

def save_config():
    global config, args

    with open(args.config, 'w') as f:
        json.dump(config, f)

def scatter_peaks(hist, config):
    peaks_x = []
    peaks_y = []
    for i in range(len(hist)):
        _, peaks = process_line(hist[i,:], config)
        line_y = [i for _ in peaks]
        peaks_x += list(peaks)
        peaks_y += line_y

    return peaks_x, peaks_y

def gui():
    global last_bin, args
    def update_image(_):
        hist_shape = f[keys[cv2.getTrackbarPos('bins', 'Histogram lines')]].shape
        cv2.setTrackbarMax('range start x', 'Histogram lines', hist_shape[1]-1)
        cv2.setTrackbarPos('range start x', 'Histogram lines', min(cv2.getTrackbarPos('range start x', 'Histogram lines'), hist_shape[1]-1))
        cv2.setTrackbarMax('range stop x', 'Histogram lines', hist_shape[1]-1)
        cv2.setTrackbarPos('range stop x', 'Histogram lines', min(cv2.getTrackbarPos('range stop x', 'Histogram lines'), hist_shape[1]-1))
        cv2.setTrackbarMax('range start y', 'Histogram lines', hist_shape[0]-1)
        cv2.setTrackbarPos('range start y', 'Histogram lines', min(cv2.getTrackbarPos('range start y', 'Histogram lines'), hist_shape[0]-1))
        cv2.setTrackbarMax('range stop y', 'Histogram lines', hist_shape[0]-1)
        cv2.setTrackbarPos('range stop y', 'Histogram lines', min(cv2.getTrackbarPos('range stop y', 'Histogram lines'), hist_shape[0]-1))
        update(42)

    def update_line(event, x, y, flags, param):
        global mx, my, scale_x, scale_y, last_bin

        sizex = cv2.getTrackbarPos('size x', 'Histogram lines')
        sizey = cv2.getTrackbarPos('size y', 'Histogram lines')
        rng = _range().update()
        rwidth = rng.x.stop - rng.x.start
        rheight = rng.y.stop - rng.y.start
        pwidth = sizex // 3
        pheight = sizey // 2
        dead_zone = 100 # This is in global scale
        if y > sizey // 2 or x < sizex // 3:
            lx = x % pwidth
            ly = y % pheight
            gx = int(lx * (1./scale_x)) if y < sizey // 2 else int(rng.x.start + lx * (1./(pwidth / rwidth)))
            gy = int(ly * (1./scale_y)) if y < sizey // 2 else int(rng.y.start + ly * (1./(pheight / rheight)))
            if (event == cv2.EVENT_LBUTTONDOWN):
                print ('down', gx, gy)
                mx = gx
                my = gy
            elif (event == cv2.EVENT_LBUTTONUP):
                print ('up', gx, gy)
                if abs(mx-gx) < dead_zone and abs(my-gy) < dead_zone:
                    cv2.setTrackbarPos('line', 'Histogram lines', gy)
                    print ('Set line', y, ly, gy)
                    #update(42)
                else:
                    mx, gx = sorted((mx,gx))
                    my, gy = sorted((my,gy))
                    cv2.setTrackbarPos('range start x', 'Histogram lines', mx)
                    cv2.setTrackbarPos('range stop x', 'Histogram lines', gx)
                    cv2.setTrackbarPos('range start y', 'Histogram lines', my)
                    cv2.setTrackbarPos('range stop y', 'Histogram lines', gy)
                    print ('Set bounds', mx, my, gy, gy)
                    #update(42)
            elif (event == cv2.EVENT_MBUTTONDOWN):
                print ('reset bounding box')
                hist_shape = f[keys[cv2.getTrackbarPos('bins', 'Histogram lines')]].shape
                cv2.setTrackbarPos('range start x', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop x', 'Histogram lines', hist_shape[1]-1)
                cv2.setTrackbarPos('range start y', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop y', 'Histogram lines', hist_shape[0]-1)
                #update(42)

    # TODO Only do recomputation, whenever parameters that have an effect are changed.
    def update(_): # Note: all colors are in BGR format, as this is what OpenCV uses
        global last_bin, config, scale_x, scale_y

        times = []
        if not update_config():
            return
        times.append(('init',time.time()))

        # Check if trackbar ranges should be updated
        bin = cv2.getTrackbarPos('bins', 'Histogram lines')
        hist = f[keys[bin]]
        if bin != last_bin:
            last_bin = bin
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
        rng = _range().update()
        selected_line = cv2.getTrackbarPos('line', 'Histogram lines')

        times.append(('trackbar parsing',time.time()))

        partial_width = cv2.getTrackbarPos('size x', 'Histogram lines') // 3
        partial_height = cv2.getTrackbarPos('size y', 'Histogram lines') // 2
        partial_size = (partial_width, partial_height)
        scale_x = partial_width / hist.shape[1]
        scale_y = partial_height / hist.shape[0]

        colored = process_with_box(hist, rng, selected_line, scale_x, scale_y, partial_size)

        times.append(('processed with box',time.time()))

        # Show the plot of a single line
        line_plot = plot_line(hist[selected_line, :], rng)
        lp_resized = cv2.resize(line_plot, partial_size)

        times.append(('plot line',time.time()))

        scatter_plot, py, px = process_scatter_peaks(hist, rng, peaks=args.peaks)
        sp_resized = cv2.resize(scatter_plot, partial_size)

        times.append(('scatter peaks',time.time()))

        first_row = np.concatenate((colored, lp_resized, sp_resized), axis=1)

        local_scale_y = partial_height / (rng.y.stop - rng.y.start)
        mask = np.zeros((rng.y.stop-rng.y.start,rng.x.stop-rng.x.start), dtype=np.uint8)
        mask[py, px] = 255

        dilated, eroded = process_closing(mask, config)
        display_eroded = cv2.resize(eroded, partial_size)
        display_eroded = cv2.cvtColor(display_eroded, cv2.COLOR_GRAY2BGR)
        if selected_line > rng.y.start and selected_line < rng.y.stop:
            display_eroded[int((selected_line-rng.y.start)*local_scale_y),:] = (0,0,255)

        times.append(('closing',time.time()))

        # Find lines
        labeled, labels = process_contours(eroded, rng, config)
        label_colours = [(0,0,0)] + [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in labels]
        labeled_colour = np.zeros((labeled.shape[0], labeled.shape[1], 3), dtype=np.uint8)
        for l in labels:
            labeled_colour[labeled == l] = label_colours[l]
        display_contours = cv2.resize(labeled_colour, partial_size)
        if selected_line > rng.y.start and selected_line < rng.y.stop:
            display_contours[int((selected_line-rng.y.start)*local_scale_y),:] = (0,0,255)

        times.append(('find lines',time.time()))

        # TODO isn't currently used
        # Find joints
        #joints = process_joints(eroded)
        #display_joints = cv2.resize(joints, partial_size)
        #display_joints = cv2.cvtColor(display_joints, cv2.COLOR_GRAY2BGR)
        #if selected_line > rng.y.start and selected_line < rng.y.stop:
        #    display_joints[int((selected_line-rng.y.start)*local_scale_y),:] = (0,0,255)

        times.append(('find joints',time.time()))

        second_row = np.concatenate((display_eroded, display_contours, display_contours), axis=1)

        cv2.imshow('Histogram lines', np.concatenate((first_row, second_row)))

        for i in range(1, len(times)):
            label, tim = times[i]
            print (label, tim - times[i-1][1])

    def update_config():
        global config

        changed = False
        for entry in config.keys():
            tmp = config[entry]
            config[entry] = cv2.getTrackbarPos(entry, 'Histogram lines')
            changed |= tmp == config[entry]

        return changed

    f = np.load(args.histogram[0])
    keys = [key for key in f.keys()]
    print ('keys', keys)
    first_hist = f[keys[0]]
    last_bin = 0
    cv2.namedWindow('Histogram lines')
    cv2.createTrackbar('range start x', 'Histogram lines', 0, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range stop x', 'Histogram lines', first_hist.shape[1]-1, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range start y', 'Histogram lines', 0, first_hist.shape[0]-1, update)
    cv2.createTrackbar('range stop y', 'Histogram lines', first_hist.shape[0]-1, first_hist.shape[0]-1, update)
    cv2.createTrackbar('size x', 'Histogram lines', 1024, 1920, update)
    cv2.createTrackbar('size y', 'Histogram lines', 512, 1080, update)
    cv2.createTrackbar('bins', 'Histogram lines', 0, len(keys)-1, update_image)
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
    update(42)

    while True:
        key = cv2.waitKey(16)
        key &= 0xFF
        if key == 113:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    if args.batch:
        batch()
    else:
        # Preload pyplot, as the first figure takes around 3 seconds to compute. For timing only
        plt.figure()
        plt.clf()
        gui()
        if not args.dry_run:
            save_config()