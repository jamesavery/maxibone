import cv2
import numpy as np
from numpy.lib.function_base import disp
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import json
# TODO Make program to run gui > dump config > run from cmd on multiple.
import argparse
import os

def load_config(filename):
    if os.path.exists(filename):
        with open('filename', 'r') as f:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Computes the connected lines in a 2D histogram. It can either be run in GUI mode, where one tries to find the optimal configuration parameters, or batch mode, where it processes one or more histograms into images.")

    parser.add_argument('histogram', nargs='+',
        help='Specifies one or more histogram files (usually *.npz) to process.')
    parser.add_argument('-b', 'batch', action='store_true',
        help='Toggles whether the script should be run in batch mode. In this mode, the GUI isn\'t launched, but the provided histograms will be stored in the specified output folder.')
    parser.add_argument('-c', 'config', default='config.json', type=str,
        help='The configuration file storing the parameters. If in GUI mode, this will be overwritten with the values on the trackbars. If it doesn\'t exist, the default values inside this script will be used.')
    parser.add_argument('-d', 'dry_run', action='store_true',
        help='Toggles whether the configuration should be saved, when running in GUI mode.')
    parser.add_argument('-o', 'output', default='output', type=str,
        help='Specifies the folder to put the resulting images in.')
    parser.add_argument('-v', 'verbose', action='store_true',
        help='Toggles whether debug printing should be enabled.')

    args = parser.parse_args()
    return args

def process_histogram(hist, closing_kernel=(10,2), iter_dilate=10, iter_erode=5, k=10, height=1000):
    # Find the peaks
    px, py = scatter_peaks(hist, k, height)

    # Create the mask
    mask = np.zeros(hist.shape, dtype=np.uint8)
    mask[py,px] = 255

    # Dilate and erode
    kernel = np.ones(closing_kernel, np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iter_dilate)
    eroded = cv2.erode(dilated, kernel, iterations=iter_erode)

    # Group and label the lines in the mask
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(hist.shape, np.uint8)
    for i in np.arange(len(contours)):
        result = cv2.drawContours(result, contours, i, int(i)+1, -1)

    return result

def process_line(line, k=10, height=1000):
    meaned = gaussian_filter1d(line, k)
    peaks, _ = signal.find_peaks(meaned, height)
    return meaned, peaks

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f)

def scatter_peaks(hist,k=10,height=1000):
    peaks_x = []
    peaks_y = []
    for i in range(len(hist)):
        _, peaks = process_line(hist[i,:], k, height)
        line_y = [i for _ in peaks]
        peaks_x += list(peaks)
        peaks_y += line_y
    return peaks_x, peaks_y

def show_gui(filename):
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
        global mx, my, scale_x, scale_y
        sizex = cv2.getTrackbarPos('size x', 'Histogram lines')
        sizey = cv2.getTrackbarPos('size y', 'Histogram lines')
        rx0 = cv2.getTrackbarPos('range start x', 'Histogram lines')
        rx1 = cv2.getTrackbarPos('range stop x', 'Histogram lines')
        ry0 = cv2.getTrackbarPos('range start y', 'Histogram lines')
        ry1 = cv2.getTrackbarPos('range stop y', 'Histogram lines')
        rwidth = rx1 - rx0
        rheight = ry1 - ry0
        pwidth = sizex // 3
        pheight = sizey // 2
        dead_zone = 100 # TODO this is in global scale...
        if y > sizey // 2 or x < sizex // 3:
            lx = x % pwidth
            ly = y % pheight
            gx = int(lx * (1./scale_x)) if y < sizey // 2 else int(rx0 + lx * (1./(pwidth / rwidth)))
            gy = int(ly * (1./scale_y)) if y < sizey // 2 else int(ry0 + ly * (1./(pheight / rheight)))
            if (event == cv2.EVENT_LBUTTONDOWN):
                print ('down', gx, gy)
                mx = gx
                my = gy
            elif (event == cv2.EVENT_LBUTTONUP):
                print ('up', gx, gy)
                if abs(mx-gx) < dead_zone and abs(my-gy) < dead_zone:
                    cv2.setTrackbarPos('line', 'Histogram lines', gy)
                    print ('Set line', y, ly, gy)
                    update(42)
                else:
                    mx, gx = sorted((mx,gx))
                    my, gy = sorted((my,gy))
                    # TODO sort positions, so that you can draw box in both directions
                    cv2.setTrackbarPos('range start x', 'Histogram lines', mx)
                    cv2.setTrackbarPos('range stop x', 'Histogram lines', gx)
                    cv2.setTrackbarPos('range start y', 'Histogram lines', my)
                    cv2.setTrackbarPos('range stop y', 'Histogram lines', gy)
                    print ('Set bounds', mx, my, gy, gy)
                    update(42)
            elif (event == cv2.EVENT_MBUTTONDOWN):
                print ('reset bounding box')
                hist_shape = f[keys[cv2.getTrackbarPos('bins', 'Histogram lines')]].shape
                cv2.setTrackbarPos('range start x', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop x', 'Histogram lines', hist_shape[1]-1)
                cv2.setTrackbarPos('range start y', 'Histogram lines', 0)
                cv2.setTrackbarPos('range stop y', 'Histogram lines', hist_shape[0]-1)
                update(42)

    # TODO Only do recomputation, whenever parameters that have an effect are changed.
    def update(_): # Note: all colors are in BGR format, as this is what OpenCV uses
        global scale_x, scale_y, last_bin

        # Check if trackbar ranges should be updated
        bin = cv2.getTrackbarPos('bins', 'Histogram lines')
        hist = f[keys[bin]]
        if bin != last_bin:
            last_bin = bin
            cv2.setTrackbarMax('range start x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarMax('range stop x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarMax('range start y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarMax('range stop y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarMax('line', 'Histogram lines', hist.shape[0]-1)

            cv2.setTrackbarPos('range start x', 'Histogram lines', 0)
            cv2.setTrackbarPos('range stop x', 'Histogram lines', hist.shape[1]-1)
            cv2.setTrackbarPos('range start y', 'Histogram lines', 0)
            cv2.setTrackbarPos('range stop y', 'Histogram lines', hist.shape[0]-1)
            cv2.setTrackbarPos('line', 'Histogram lines', 0)

        # Show the original image, along with the line
        range_x_start = cv2.getTrackbarPos('range start x', 'Histogram lines')
        range_x_stop  = cv2.getTrackbarPos('range stop x', 'Histogram lines')
        range_y_start = cv2.getTrackbarPos('range start y', 'Histogram lines')
        range_y_stop  = cv2.getTrackbarPos('range stop y', 'Histogram lines')
        selected_line = cv2.getTrackbarPos('line', 'Histogram lines')
        hist_sum = np.sum(hist, axis=1)
        hist_sum[hist_sum==0] = 1
        hist = hist / hist_sum[:,np.newaxis]
        #disp_hist = hist/hist_sum[:,np.newaxis]
        disp_hist = hist
        display = ((disp_hist.astype(np.float32) / disp_hist.max()) * 255.0).astype(np.uint8)
        partial_width = cv2.getTrackbarPos('size x', 'Histogram lines') // 3
        partial_height = cv2.getTrackbarPos('size y', 'Histogram lines') // 2
        partial_size = (partial_width, partial_height)
        scale_x = partial_width / display.shape[1]
        scale_y = partial_height / display.shape[0]
        #hist = hist[range_y_start:range_y_stop,range_x_start:range_x_stop]
        box_x_start = int(range_x_start * scale_x)
        box_y_start = int(range_y_start * scale_y)
        box_x_stop = int(range_x_stop * scale_x)
        box_y_stop = int(range_y_stop * scale_y)
        resized = cv2.resize(display, partial_size)
        colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        colored = cv2.rectangle(colored, (box_x_start, box_y_start), (box_x_stop, box_y_stop), (0, 255, 0), 1)
        colored[int(selected_line*scale_y),:] = (0,0,255)

        # Show the plot of a single line
        line = disp_hist[selected_line, :]
        min_height = .01*cv2.getTrackbarPos('min peak height', 'Histogram lines')*line.max()
        meaned, peaks = process_line(line, cv2.getTrackbarPos('smooth', 'Histogram lines'), min_height)
        fig, ax = plt.subplots()
        #ax.axvspan(range_x_start, range_x_stop, color='green', alpha=0.1)
        ax.plot(line[range_x_start:range_x_stop])
        ax.plot(meaned[range_x_start:range_x_stop])
        #peaks_filt = [peak for peak in peaks if peak > range_x_start and peak < range_
        peaks_filt = peaks[(peaks > range_x_start) & (peaks < range_x_stop)] - range_x_start

        ax.scatter(peaks_filt, meaned[peaks_filt], c='red')
        line_plot = mplfig_to_npimage(fig)
        line_plot = cv2.cvtColor(line_plot, cv2.COLOR_RGB2BGR)
        plt.close()
        lp_resized = cv2.resize(line_plot, partial_size)

        # Show the peaks scattered on the original
        px = []
        py = []
        for i in np.arange(range_y_stop-range_y_start):
            line = hist[i+range_y_start,:]
            min_height = .01*cv2.getTrackbarPos('min peak height', 'Histogram lines')*line.max()
            _, peaks = process_line(line, cv2.getTrackbarPos('smooth', 'Histogram lines'), min_height)
            peaks = [peak-range_x_start for peak in peaks if peak > range_x_start and peak < range_x_stop]
            py += [i for _ in np.arange(len(peaks))]
            px += list(peaks)
        fig, ax = plt.subplots()
        ax.imshow(hist[range_y_start:range_y_stop,range_x_start:range_x_stop], cmap='jet')
        # TODO lav opacity
        #ax.scatter(px, py)
        scatter_plot = mplfig_to_npimage(fig)
        scatter_plot = cv2.cvtColor(scatter_plot, cv2.COLOR_RGB2BGR)
        plt.close()
        sp_resized = cv2.resize(scatter_plot, partial_size)

        first_row = np.concatenate((colored, lp_resized, sp_resized), axis=1)

        local_scale_y = partial_height / (range_y_stop - range_y_start)
        mask = np.zeros((range_y_stop-range_y_start,range_x_stop-range_x_start), dtype=np.uint8)
        mask[py, px] = 255

        close_kernel = np.ones((cv2.getTrackbarPos('close kernel y', 'Histogram lines'), cv2.getTrackbarPos('close kernel x', 'Histogram lines')))
        dilated = cv2.dilate(mask, close_kernel, iterations=cv2.getTrackbarPos('iter dilate', 'Histogram lines'))
        #display_dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        #display_dilated = cv2.resize(display_dilated, partial_size)
        #if selected_line > range_y_start and selected_line < range_y_stop:
        #    display_dilated[int((selected_line-range_y_start)*local_scale_y),:] = (0,0,255)

        eroded = cv2.erode(dilated, close_kernel, iterations=cv2.getTrackbarPos('iter erode', 'Histogram lines'))
        display_eroded = cv2.resize(eroded, partial_size)
        display_eroded = cv2.cvtColor(display_eroded, cv2.COLOR_GRAY2BGR)
        if selected_line > range_y_start and selected_line < range_y_stop:
            display_eroded[int((selected_line-range_y_start)*local_scale_y),:] = (0,0,255)

        # Find joints
        #joint_kernel_size = cv2.getTrackbarPos('joint kernel size', 'Histogram lines')
        #horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (joint_kernel_size,1))
        #horizontal = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        #vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,joint_kernel_size))
        #vertical = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        #joints = cv2.bitwise_and(horizontal, vertical)
        #display_joints = cv2.resize(joints, partial_size)
        #display_joints = cv2.cvtColor(display_joints, cv2.COLOR_GRAY2BGR)
        #if selected_line > range_y_start and selected_line < range_y_stop:
        #    display_joints[int((selected_line-range_y_start)*local_scale_y),:] = (0,0,255)

        # Find lines
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros((range_y_stop-range_y_start, range_x_stop-range_x_start, 3), np.uint8)
        for i in np.arange(len(contours)):
            if cv2.contourArea(contours[i]) > cv2.getTrackbarPos('min contour size', 'Histogram lines'):
                result = cv2.drawContours(result, contours, i, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), -1)
        display_result = cv2.resize(result, partial_size)
        if selected_line > range_y_start and selected_line < range_y_stop:
            display_result[int((selected_line-range_y_start)*local_scale_y),:] = (0,0,255)

        skeleton = skeletonize(result)
        skeleton_gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        skeleton_gray = cv2.threshold(skeleton_gray, 1, 255, cv2.THRESH_BINARY)[1]
        skeleton_gray = cv2.dilate(skeleton_gray, (3,3), 10)
        joint_kernel_size = cv2.getTrackbarPos('joint kernel size', 'Histogram lines')
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (joint_kernel_size,1))
        horizontal = cv2.morphologyEx(skeleton_gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,joint_kernel_size))
        vertical = cv2.morphologyEx(skeleton_gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        joints = cv2.bitwise_and(horizontal, vertical)
        joints = cv2.dilate(joints, (10,10), iterations=5)
        display_joints = cv2.resize(joints, partial_size)
        display_joints = cv2.cvtColor(display_joints, cv2.COLOR_GRAY2BGR)
        if selected_line > range_y_start and selected_line < range_y_stop:
            display_joints[int((selected_line-range_y_start)*local_scale_y),:] = (0,0,255)

        #second_row = np.concatenate((display_dilated, display_eroded, display_result), axis=1)
        #second_row = np.concatenate((display_eroded, display_result, display_joints), axis=1)
        second_row = np.concatenate((display_eroded, display_result, display_joints), axis=1)


        cv2.imshow('Histogram lines', np.concatenate((first_row, second_row)))

    f = np.load(filename)
    keys = [key for key in f.keys()]
    first_hist = f[keys[0]]
    cv2.namedWindow('Histogram lines')
    cv2.createTrackbar('range start x', 'Histogram lines', 0, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range stop x', 'Histogram lines', first_hist.shape[1]-1, first_hist.shape[1]-1, update)
    cv2.createTrackbar('range start y', 'Histogram lines', 0, first_hist.shape[0]-1, update)
    cv2.createTrackbar('range stop y', 'Histogram lines', first_hist.shape[0]-1, first_hist.shape[0]-1, update)
    cv2.createTrackbar('size x', 'Histogram lines', 1024, 1920, update)
    cv2.createTrackbar('size y', 'Histogram lines', 512, 1080, update)
    cv2.createTrackbar('bins', 'Histogram lines', 0, len(keys)-1, update_image)
    cv2.createTrackbar('line', 'Histogram lines', 0, first_hist.shape[0]-1, update)
    cv2.createTrackbar('smooth', 'Histogram lines', 7, 50, update)
    cv2.createTrackbar('min peak height', 'Histogram lines', 1, 100, update)
    cv2.createTrackbar('close kernel x', 'Histogram lines', 2, 100, update)
    cv2.createTrackbar('close kernel y', 'Histogram lines', 10, 100, update)
    cv2.createTrackbar('iter dilate', 'Histogram lines', 10, 100, update)
    cv2.createTrackbar('iter erode', 'Histogram lines', 5, 100, update)
    cv2.createTrackbar('min contour size', 'Histogram lines', 2000, 10000, update)
    cv2.createTrackbar('joint kernel size', 'Histogram lines', 2, 100, update)
    cv2.setMouseCallback('Histogram lines', update_line)
    update(42)
    cv2.waitKey(0) # Segfaults for some reason :) Looks like it's some wayland / ubuntu 20.04 error
    #while True:
    #    key = cv2.waitKey(16)
    #    key &= 0xFF
    #    if key == 113:
    #        break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    if args.batch:
        batch()
    else:
        gui(config)
        if not args.dry_run:
            save_config(config)
    sample = '770c_pag'
    filename = f"/mnt/data/MAXIBONE/Goats/tomograms/processed/histograms/{sample}/bins1.npz"
    mx, my = 0, 0
    last_bin = 0
    show_gui(filename)
