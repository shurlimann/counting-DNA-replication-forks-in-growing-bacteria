import numpy as np
import pandas as pd

# Plotting for debug
import matplotlib as mpl
from matplotlib import pyplot as plt

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}
mpl.rc('font', **font)
mpl.rcParams['pdf.fonttype'] = 42

import skimage as sk
from skimage import registration, morphology, measure, io
from skimage.transform import warp, AffineTransform
from skimage.feature import blob_log
from skimage.filters import threshold_triangle, gaussian
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

from scipy.optimize import leastsq  # fitting 2d gaussian
from scipy.ndimage.measurements import histogram

import os
import shutil
import tifffile
from nd2reader import ND2Reader
import cv2 as cv
import yaml


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    if params are not provided, they are calculated from the moments
    params should be (height, x, y, width_x, width_y)"""
    gparams = moments(data)  # create guess parameters.
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorfunction, gparams)
    return p, success


def moments(data):
    '''
    Returns (height, x, y, width_x, width_y)
    The (circular) gaussian parameters of a 2D distribution by calculating its moments.
    width_x and width_y are 2*sigma x and sigma y of the gaussian.
    '''
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width = float(np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum()))
    row = data[int(x), :]
    # width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width


def gaussian(height, center_x, center_y, width):
    '''Returns a gaussian function with the given parameters. It is a circular gaussian.
    width is 2*sigma x or y
    '''
    # return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    return lambda x, y: height * np.exp(-(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2)


def create_circular_mask(shape, center=None, radius=None):
    """
    Given the height and width of an image, and the center and radius of a circular mask, returns a circular mask.
    """
    (h, w) = shape
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask.astype(int)


def foci_lap(img_phase, img_mask, img_foci, cell, params, fov, debug_foci=None):
    '''foci_lap finds foci using a laplacian convolution then fits a 2D
    Gaussian.

    The returned information are the parameters of this Gaussian.
    All the information is returned as a pandas dataframe.

    Adapted from https://github.com/junlabucsd/mm3/blob/master/mm3_helpers.py

    Parameters
    ----------
    img_phase : 2D np.array
        phase contract or bright field image.
    img_mask : 2D np.array
        mask of cells where individual cells are instance segmented
    img_foci : 2D np.array
        fluorescent image with foci.
    cell : cell object from skimage.measure.regionprops
    params : dict


    Returns
    -------
    spots : pd.DataFrame
    '''

    # pull out useful information for just this time point
    bbox = cell.bbox  # es[i]
    orientation = cell.orientation  # s[i]
    centroid = cell.centroid  # s[i]
    region = cell.label  # s[i]
    coords = cell.coords

    # declare arrays which will hold foci data
    disp_l = []  # displacement in length of foci from cell center
    disp_w = []  # displacement in width of foci from cell center
    foci_a_fixed = []  # foci total amount (from raw image)
    foci_a_radius = []  # foci total amount (based on blob_log radius)
    foci_a_circ = []  # foci total amount (based on a circular mask)
    foci_h = []  # foci height (based on Gaussian fit)
    foci_fit = []  # how well the Gaussian fits the foci
    max_intensity = []

    # define parameters for foci finding
    minsig = params['foci']['foci_log_minsig']  # minimum size of spot
    maxsig = params['foci']['foci_log_maxsig']  # maximum size of spot
    thresh = params['foci']['foci_log_thresh']  # threshold in which laplacian well reaches to record potential foci
    peak_med_ratio = params['foci']['foci_log_peak_med_ratio']  #
    if debug_foci is None:
        debug_foci = params['foci']['debug_foci']
    if debug_foci:
        print(region)
    over_lap = params['foci']['over_lap']  # if two blobs overlap by more than this fraction, smaller blob is cut
    cell_subtract = params['foci']['cell_subtract']

    # calculate median intensity of the fluorescent channel in the cell. used to filter foci
    img_foci_masked = np.copy(img_foci).astype(np.float)
    img_foci_masked[img_mask != region] = np.nan
    cell_fl_median = np.nanmedian(img_foci_masked)

    img_foci_masked[img_mask != region] = 0

    # subtract the median intensity from the fluorescent channel in the cell
    img_foci_raw = img_foci.astype('int32')
    if cell_subtract:
        img_foci = img_foci_raw.astype('int32') - cell_fl_median.astype('int32')
        img_foci[img_foci < 0] = 0
        img_foci = img_foci.astype('uint16')

    # find blobs using Laplacian of gaussian
    numsig = int(2 * (maxsig - minsig + 1))  # number of radii to consider between min ang max sig
    blobs = blob_log(img_foci_masked,
                     min_sigma=minsig,
                     max_sigma=maxsig,
                     overlap=over_lap,
                     num_sigma=numsig,
                     threshold=thresh)

    # these will hold information about foci position temporarily
    x_blob, y_blob, r_blob = [], [], []  # information about the blobs from blob_log in absolute coordinates
    x_gaus, y_gaus, w_gaus = [], [], []  # information about the peaks from fitting a gaussian

    # loop through each potential foci
    if debug_foci:
        print(f'initial number of blobs found: {np.shape(blobs)[0]}')
    for blob in blobs:
        y_loc, x_loc, sig = blob  # x location, y location, and sigma of gaus
        x_loc = int(np.around(x_loc))  # switch to int for slicing images
        y_loc = int(np.around(y_loc))
        radius_exact = np.sqrt(2) * sig
        radius = int(np.ceil(np.sqrt(2) * sig))  # will be used to slice out area around foci

        if (y_loc, x_loc) in list(map(tuple, coords)):  # if the blob is in the cell

            # cut out a small image from original image to fit gaussian
            gfit_area = img_foci[y_loc - radius:y_loc + radius,
                        x_loc - radius:x_loc + radius]

            maxsig_int = int(np.ceil(maxsig))
            gfit_area_fixed = img_foci[y_loc - maxsig_int:y_loc + maxsig_int,
                              x_loc - maxsig_int:x_loc + maxsig_int]

            # fit gaussian to proposed foci in small box
            if np.sum(gfit_area) > 0:
                p, success = fitgaussian(gfit_area)
                (peak_fit, x_fit, y_fit, w_fit) = p
            else:
                (peak_fit, x_fit, y_fit, w_fit) = (0, 0, 0, 0)

            # print('peak', peak_fit)
            if x_fit <= 0 or x_fit >= radius * 2 or y_fit <= 0 or y_fit >= radius * 2:
                if debug_foci: print('Throw out foci (gaus fit not in gfit_area)')
                continue
            elif peak_fit / (cell_fl_median + 1) < peak_med_ratio:
                if debug_foci: print(f'Peak ({peak_fit}) does not pass height test.')
                continue
            else:
                x_blob.append(x_loc)  # for plotting
                y_blob.append(y_loc)  # for plotting
                r_blob.append(radius)
                foci_fit.append(success)

                # find x and y position relative to the whole image (convert from small box)
                x_rel = x_loc - radius + x_fit
                y_rel = y_loc - radius + y_fit
                x_gaus = np.append(x_gaus, x_rel)  # for plotting
                y_gaus = np.append(y_gaus, y_rel)  # for plotting
                w_gaus = np.append(w_gaus, w_fit)  # for plotting, width (variance) of the gaussian

                # calculate distance of foci from middle of cell (scikit image)
                if orientation < 0:
                    orientation = np.pi + orientation
                disp_y = (y_rel - centroid[0]) * np.sin(orientation) - (x_rel - centroid[1]) * np.cos(orientation)
                disp_x = (y_rel - centroid[0]) * np.cos(orientation) + (x_rel - centroid[1]) * np.sin(orientation)

                # calculate the total fluorescence using a circular mask
                circ_mask = create_circular_mask(shape=np.shape(img_foci),
                                                 center=[x_loc, y_loc],
                                                 radius=radius_exact)
                foci_a_circ = np.append(foci_a_circ,
                                        np.sum(circ_mask * img_foci))
                max_intensity = np.append(max_intensity,
                                          np.max(circ_mask * img_foci))

                # append foci information to the list
                disp_l = np.append(disp_l, disp_y)
                disp_w = np.append(disp_w, disp_x)
                foci_a_fixed = np.append(foci_a_fixed, np.sum(gfit_area_fixed))
                foci_a_radius = np.append(foci_a_radius, np.sum(gfit_area))
                foci_h = np.append(foci_h, peak_fit)
        else:
            if debug_foci:
                print('Blob not in bounding box.')
                print(x_loc, y_loc)

    # store information about all the spots in the given cell
    spots = pd.DataFrame()
    spots['x_pos_gaussian'] = x_gaus
    spots['y_pos_gaussian'] = y_gaus
    spots['r_gaussian'] = w_gaus
    # spots['gaussian_fit'] = foci_fit
    spots['x_pos_blob'] = x_blob
    spots['y_pos_blob'] = y_blob
    spots['r_blob'] = r_blob
    spots['foci_area_fixed'] = foci_a_fixed
    spots['foci_area_radius'] = foci_a_radius
    spots['foci_area_circular'] = foci_a_circ
    spots['gaussian_height'] = foci_h
    spots['max_intensity'] = max_intensity
    spots['cell_median_fluorescence'] = cell_fl_median
    spots['cell_id'] = region
    spots['cell_size'] = np.size(coords) / 2
    spots['background_subtract'] = cell_subtract
    spots['FOV'] = fov
    # spots['solidity'] = cell.solidity

    spots['peak_med_ratio'] = spots.gaussian_height / spots.cell_median_fluorescence
    spots['max_med_ratio'] = spots.max_intensity / spots.cell_median_fluorescence
    spots['spot_id'] = [f'f{fov}_c{region}_s{s + 1}' for s in range(len(x_gaus))]

    if debug_foci:  # generating figures of phase images, the mask, and the fluorescence with detected spots
        base_dir = params['paths']['base_dir']
        debug_dir = params['foci']['debug_subdir']
        output_dir = f'{base_dir}/{debug_dir}'

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        buffer = 10
        ymin = max(0, bbox[0] - buffer)
        ymax = min(np.shape(img_mask)[0], bbox[2] + buffer)
        xmin = max(0, bbox[1] - buffer)
        xmax = min(np.shape(img_mask)[1], bbox[3] + buffer)

        fig = plt.figure(figsize=((3 * (xmax - xmin) + 15) / 6,
                                  (ymax - ymin) / 6))

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(img_phase[ymin: ymax, xmin: xmax])
        ax.set_title('phase')

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow((img_mask == region)[ymin: ymax, xmin: xmax])
        ax.set_title('mask')

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(img_foci_raw[ymin: ymax, xmin: xmax])
        ax.set_title('fluorescence')

        for index, spot in spots.iterrows():
            x_gauss = spot['x_pos_gaussian'] - xmin
            y_gauss = spot['y_pos_gaussian'] - ymin
            r_gauss = spot['r_gaussian'] * np.sqrt(2)
            c_gauss = plt.Circle((x_gauss, y_gauss),
                                 r_gauss,
                                 color='blue',
                                 linewidth=2,
                                 fill=False,
                                 alpha=0.5)
            # ax.add_patch(c_gauss)

            x_blob = spot['x_pos_blob'] - xmin
            y_blob = spot['y_pos_blob'] - ymin
            r_blob = spot['r_blob'] * np.sqrt(2)

            if ((spot.peak_med_ratio > 1.8) or (spot.r_gaussian < 1.8)) \
                    and (spot.cell_median_fluorescence > 100) \
                    and (spot.r_gaussian > 1):
                r_color = 'white'
            else:
                r_color = 'red'
            c_blob = plt.Circle((x_blob, y_blob),
                                r_blob,
                                color=r_color,
                                linewidth=2,
                                fill=False,
                                alpha=0.5)
            ax.add_patch(c_blob)

        fig.suptitle(f'FOV {fov}, cell #{region}', fontsize=30)
        plt.subplots_adjust(wspace=0.001)
        fig.savefig(f'{output_dir}/FOV{fov}_cell{region}.png',
                    bbox_inches='tight')
        plt.close()

    return spots


def find_cells(im, thresh):
    """
    Given a phase image of bacterial cells, finds where the cells are.
    Args:
        im (2-D numpy array): Phase image of cells
        thresh (float, optional): how much darker the cells are than the background 
        (theoretically exposure invariant)
    Returns:
        mask (2-D numpy array): labelled mask of the cells 
    """
    # Background subtracts and inverts the image so that darker regions (i.e. the cells
    # are brighter). Then finds where the cells are by finding the brightest peaks
    back_sub = -1 * im + np.median(im)
    foreground = back_sub > thresh * 255
    background = (back_sub < thresh * 255)

    background = background.astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    background = cv.erode(background, kernel)
    cells = sk.measure.label(foreground)

    # removes anything that is smaller than 125 pixels
    hist_cells = histogram(cells, 0, np.max(cells), np.max(cells))
    small_cells = np.where(hist_cells < 125)[0]
    icells = np.isin(cells, small_cells)
    cells[icells] = 0
    markers = foreground * (cells + 1) + background

    # Finds the edges using a Scharr filter and smooths with a Gaussian filter
    edges_x = cv.Scharr(im, cv.CV_64F, 1, 0)
    edges_y = cv.Scharr(im, cv.CV_64F, 0, 1)
    edges = np.uint16(np.sqrt(np.square(edges_x) + np.square(edges_y)))
    edges = cv.GaussianBlur(edges, (3, 3), 0)

    markers = watershed(edges, markers)
    mask = markers != 1

    hist_mask = histogram(markers, 0.5, np.max(markers) + 0.5, np.max(markers))
    small_cells = np.where(hist_mask < 125)[0]
    if np.size(small_cells) > 0:
        icells = np.isin(markers, small_cells + 1)
        mask[icells] = 0

    return mask


def register_images(param_file_path):
    """
    Given a file path to a yaml parameters file, registers phase and any 
    fluorescent channels and removes any registered images that were deemed
    inappropriate or require unusually large translations to register. The 
    distance translated to register the image is saved as a csv file.
    Args:
        param_file_path (string): path name to the parameter file
    Returns:
        None
    """
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)

    base_dir = params['paths']['base_dir']
    nd2_file_name = params['paths']['foci_file_path']
    registered_images_subdir = params['paths']['registered_images_subdir']
    channels = params['channels']

    if not os.path.isdir(f'{base_dir}/{registered_images_subdir}'):
        os.mkdir(f'{base_dir}/{registered_images_subdir}')

    file = f'{base_dir}/{nd2_file_name}'

    images = ND2Reader(file)
    images.iter_axes = 'vtzc'
    large_offset = []
    translations = pd.DataFrame(dtype=float,
                                index=range(1, images.sizes['v'] + 1),
                                columns=range(1, images.sizes['c']))
    for v in images.metadata['fields_of_view']:
        im = np.array(images[v * images.sizes['c']: (v + 1) * images.sizes['c']])
        offset_im = np.zeros((images.sizes['c'],
                              images.sizes['y'],
                              images.sizes['x']))
        offset_im[0, :, :] = im[0, :, :]
        for c in range(images.sizes['c'] - 1):
            translation = registration.phase_cross_correlation(sk.util.invert(im[0, :, :]),
                                                               im[c + 1, :, :],
                                                               reference_mask=np.ones((images.sizes['y'],
                                                                                       images.sizes['x'])),
                                                               upsample_factor=8,
                                                               return_error=False,
                                                               overlap_ratio=0.99)
            translations.loc[(int(v + 1)), int(c + 1)] = np.linalg.norm(translation)
            if np.linalg.norm(translation) > 10:
                large_offset = np.append(large_offset, v + 1)
                # print(f'Large offset for frame {v + 1}: {np.linalg.norm(translation)}')
            tform = AffineTransform(translation=-1 * translation[::-1])
            offset_im[c + 1, :, :] = warp(im[c + 1, :, :], tform,
                                          mode='constant',
                                          cval=0,
                                          preserve_range=True)

        # for saving each channel individually
        for c in range(images.sizes['c']):
            tifffile.imwrite('{base_dir}/{subdir}/{filename}_xy{n}_{channel}.tif'.format(base_dir=base_dir,
                                                                                         subdir=registered_images_subdir,
                                                                                         filename=file.split('/')[-1][
                                                                                                  :-4],
                                                                                         n=v + 1,
                                                                                         channel=channels[c]),
                             data=offset_im[c, :, :].astype('uint16'),
                             imagej=True,
                             resolution=(1. / images.metadata['pixel_microns'], 1. / images.metadata['pixel_microns']),
                             metadata={'axes': 'YX',
                                       'unit': 'microns'})
        tifffile.imwrite('{base_dir}/{subdir}/{filename}_no_offset_xy{n}_c2.tif'.format(base_dir=base_dir,
                                                                                        subdir=registered_images_subdir,
                                                                                        filename=file.split('/')[-1][
                                                                                                 :-4],
                                                                                        n=v + 1),
                         data=im[1, :, :].astype('uint16'),
                         imagej=True,
                         resolution=(1. / images.metadata['pixel_microns'], 1. / images.metadata['pixel_microns']),
                         metadata={'axes': 'YX',
                                   'unit': 'microns'})
    # save the translations
    translations.to_csv(f'{base_dir}/translations.csv',
                        index_label='FOV')

    # gets rid of any frames that had a large offset when trying to register the different channels
    bad_frames = np.array(large_offset)
    bad_frames = bad_frames[~np.isnan(bad_frames)]  # remove nans
    bad_frames = np.unique(bad_frames).astype(np.int)  # only looks at unique frames
    for frame in bad_frames:
        print(f'Deleting files from frame {frame}')
        for c in range(images.sizes['c']):
            filename = file.split('/')[-1][:-4]
            channel = channels[c]
            frame_file = f'{base_dir}/{registered_images_subdir}/{filename}_xy{frame}_{channel}.tif'
            if os.path.exists(frame_file):
                os.remove(frame_file)


def segment_images(param_file_path):
    """
    Given a file path to a yaml parameters file, segments the phase images and 
    saves the masks (2-D binary array) and labeled masks (2-D int array) as 
    tif files in directories specified in the parameters file. 
    Args:
        param_file_path (string): path name to the parameter file
    Returns:
        None
    """
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)

    base_dir = params['paths']['base_dir']
    labeled_mask_subdir = params['paths']['labeled_mask_subdir']
    mask_subdir = params['paths']['mask_subdir']
    nd2_file_name = params['paths']['foci_file_path']

    print(f'Segmenting the images for {nd2_file_name}')

    if not os.path.isdir(f'{base_dir}/{mask_subdir}'):
        os.mkdir(f'{base_dir}/{mask_subdir}')
    if not os.path.isdir(f'{base_dir}/{labeled_mask_subdir}'):
        os.mkdir(f'{base_dir}/{labeled_mask_subdir}')

    sigma = params['segmentation']['sigma']
    phase_im = [f'{base_dir}/split_images/{f}' \
                for f in os.listdir(f'{base_dir}/split_images') \
                if 'phase' in f]

    for im_name in phase_im:
        fov = im_name.split('_xy')[1].split('_')[0]
        im = sk.io.imread(im_name).astype(np.uint16)

        # applies a Gaussian blur to the image and then normalizes the image so that the maximum is 255
        smooth_im = cv.GaussianBlur(im, (sigma, sigma), 0)
        smooth_im = smooth_im.astype(np.float)
        smooth_im = (smooth_im - np.min(smooth_im)) / (np.percentile(smooth_im, 99.99) - np.min(smooth_im))
        smooth_im[smooth_im > 1] = 1
        smooth_im = (smooth_im * 255).astype(np.uint16)

        mask = find_cells(smooth_im, thresh=params['segmentation']['thres'])
        tifffile.imwrite(f'{base_dir}/{mask_subdir}/FOV_{fov}.tif',
                         data=mask.astype('uint16'))

        # labels the mask
        cells_labeled = label(mask).astype(np.uint16)

        # removes cells that are on or near the edge
        label_edge = np.unique(np.concatenate((cells_labeled[:, :10],
                                               cells_labeled[:, -11:],
                                               cells_labeled[:10, :],
                                               cells_labeled[-11:, :]),
                                              axis=None))
        if np.size(label_edge > 1):
            i_edge = np.isin(cells_labeled, label_edge)
            cells_labeled[i_edge] = 0
        tifffile.imwrite(f'{base_dir}/{labeled_mask_subdir}/FOV_{fov}.tif',
                         data=cells_labeled.astype('uint16'),
                         dtype=np.uint16)


def process_experiment(param_file_path, debug_foci=None):
    """
    Given a file path to a yaml parameters file, aligns and segments the phase 
    images if needed and then saves the information about the spots in a csv
    file. Note: if images are not already segmented, will segment the images
    based purely off of the phase images and classical computer vision techniques.
    This might result in undersegmentation of the image. The user can supply their
    own labeled masks by placing segmented images in 'labeled_mask_subdir'.
    Args:
        param_file_path (string): path name to the parameter file
    Returns:
        None
    """
    with open(param_file_path, 'r') as param_file:
        params = yaml.safe_load(param_file)

    # loads the appropriate directories
    base_dir = params['paths']['base_dir']
    nd2_file_path = params['paths']['foci_file_path']
    bkg_file_path = params['paths']['bkg_file_path']
    flatfield_file_path = params['paths']['flatfield_file_path']
    labeled_mask_subdir = params['paths']['labeled_mask_subdir']

    # loads the registered fluorescent and phase images along with the phase masks
    need_to_register = False
    need_to_label = False

    try:
        green_ims = [f'{base_dir}/split_images/{f}' \
                     for f in os.listdir(f'{base_dir}/split_images') \
                     if 'green' in f]
        phase_ims = [f'{base_dir}/split_images/{f}' \
                     for f in os.listdir(f'{base_dir}/split_images') \
                     if 'phase' in f]

        if len(green_ims) == 0 or len(phase_ims) == 0:
            need_to_register = True  # register if no images exist
    except FileNotFoundError:
        need_to_register = True  # register if split_images dir doesn't exist

    try:
        labeled_masks = [f'{base_dir}/{labeled_mask_subdir}/{f}' \
                         for f in os.listdir(f'{base_dir}/{labeled_mask_subdir}') \
                         if 'FOV' in f]

        if len(labeled_masks) == 0:
            need_to_label = True

    except FileNotFoundError:
        need_to_label = True

    if need_to_register:
        register_images(param_file_path)
        green_ims = [f'{base_dir}/split_images/{f}' \
                     for f in os.listdir(f'{base_dir}/split_images') \
                     if 'green' in f]
        phase_ims = [f'{base_dir}/split_images/{f}' \
                     for f in os.listdir(f'{base_dir}/split_images') \
                     if 'phase' in f]

    if need_to_label:
        segment_images(param_file_path)
        labeled_masks = [f'{base_dir}/{labeled_mask_subdir}/{f}' \
                         for f in os.listdir(f'{base_dir}/{labeled_mask_subdir}') \
                         if 'FOV' in f]
    if debug_foci is None:
        debug_foci = params['foci']['debug_foci']
    if debug_foci:  # generating figures of phase images, the mask, and the fluorescence with detected spots
        base_dir = params['paths']['base_dir']
        debug_dir = params['foci']['debug_subdir']
        output_dir = f'{base_dir}/{debug_dir}'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # loads images for flatfield correction and darknoise
    with ND2Reader(flatfield_file_path) as images:
        i_flat = np.mean(images, axis=0)

    # loads the dark noise
    with ND2Reader(bkg_file_path) as images:
        i_dark = np.mean(images, axis=(0))

    # calculates the background by taking the median of the green images
    green = ND2Reader(f'{base_dir}/{nd2_file_path}')
    green.default_coords['c'] = 1
    bkg = np.median(green, axis=0).astype(np.uint16)
    i_bkg = cv.GaussianBlur(bkg.astype(np.uint16),
                            ksize=(0, 0),
                            sigmaX=5,
                            sigmaY=5,
                            borderType=cv.BORDER_REPLICATE)

    # correction factor for flatfield correction
    norm = np.mean((i_flat - i_dark))

    # creates a list of every field of view
    fov_cells = np.stack(np.char.split(green_ims, '_xy'), axis=0)[:, 1]
    fov_cells = np.stack(np.char.split(fov_cells, '_'), axis=0)[:, 0]

    csv_name = f'{base_dir}/spots_info.csv'
    spots = pd.DataFrame()

    for fov in fov_cells:
        im_name = [f for f in green_ims if f'xy{fov}_' in f]
        mask_name = [f for f in labeled_masks if f'FOV_{fov}.tif' in f]
        phase_name = [f for f in phase_ims if f'xy{fov}_' in f]

        if len(im_name) == 1 and len(mask_name) == 1 and len(phase_name) == 1:
            img_foci = tifffile.imread(im_name[0])
            img_mask = tifffile.imread(mask_name[0])
            img_phase = tifffile.imread(phase_name[0])

            background = (np.array(img_foci) < threshold_triangle(img_foci[10:-10, 10:-10]))
            background = sk.morphology.remove_small_holes(background, area_threshold=10)

            background_fl = (background * img_foci).astype(np.float)
            background_fl[~background] = np.nan

            bkg = (i_bkg.astype(np.float) - np.mean(i_bkg)) + np.nanmean(background_fl)

            # applies a flatfield correction to the fluorescent image
            img_foci = (img_foci.astype(np.int32) - bkg) * norm / (i_flat - i_dark)
            img_foci = np.clip(img_foci, 0, np.max(img_foci))

            cells = regionprops(img_mask)
            # iterates over the cells in the labeled mask
            for cell in cells:
                s = foci_lap(img_phase, img_mask, img_foci, cell, params, fov, debug_foci)
                spots = pd.concat([spots, s])



        elif len(im_name) != 1:
            print(f'Uh oh: can\'t find a unique fluorescence image for FOV {fov}')

        elif len(mask_name) != 1:
            print(f'Uh oh: can\'t find a unique labeled mask for FOV {fov}')

    for k in params['details'].keys():
        spots[k] = params['details'][k]

    spots.to_csv(csv_name, index=False)


def open_spots_csv(param_file_paths):
    spots = pd.DataFrame()
    for param_file_path in param_file_paths[:]:
        with open(param_file_path, 'r') as param_file:
            params = yaml.safe_load(param_file)
        base_dir = params['paths']['base_dir']
        s = pd.read_csv(f'{base_dir}/spots_info.csv',
                        index_col=False)
        s.loc[:, 'base_dir'] = base_dir
        spots = pd.concat((spots, s), axis=0)
    spots = spots.reset_index(drop=True)
    return spots
# %%
