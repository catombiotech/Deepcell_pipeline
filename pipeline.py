from argv import pipe_parser
import argparse as arg

import os
import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import suffix, SortFilename

from deepcell.applications.mesmer import Mesmer as MultiplexSegmentation
import imageio
from skimage import io


from celltrack import CellTracker
import pandas as pd

def sort_file(INPUT_PATH):
    all_file = os.listdir(INPUT_PATH)
    all_file = SortFilename(all_file)
    target_file = []
    if format_list:
        for file in all_file:
            if suffix(file) in format_list:
                target_file.append(file)
    else:
        target_file = all_file
    return target_file

def load_image(INPUT_PATH):
    good_file = []
    target_file = sort_file(INPUT_PATH)
    raw_image = []
    for file in target_file:
        abs_path = os.path.join(INPUT_PATH, file)
        try:
            img = Image.open(abs_path)
            img = np.array(img)
            img = np.expand_dims(img, axis=-1)
            raw_image.append(img)
            good_file.append(file)
        except Exception as e:
            pass
        continue
    raw_image = np.array(raw_image)
    return raw_image, good_file

if __name__ == '__main__':
    
    # receive parameters from command line
    parser = arg.ArgumentParser()
    Pipe_par = pipe_parser(parser)
    args = Pipe_par.args()
    
    INPUT_PATH = args.input[0]
    OUTPUT_PATH = args.output[0]
    CSV_PATH = args.csv[0]
    image_mpp = args.mpp[0]
    
    if args.zdistance:
        z_value = args.zdistance[0]
    format_list = args.format
    if args.gif:
        gif_fps = args.gif[0]
    if args.ksize and args.ero and args.dil:
        opening_morphology_kwargs = {'kernel_size':args.ksize[0],
                                'erode_iter_number':args.ero[0],
                                'dilate_iter_number':args.dil[0]}
    else:
        opening_morphology_kwargs = None
    
    # load images
    raw_image, good_file = load_image(INPUT_PATH)
    
    # Get batch size and image size
    batch_size = raw_image.shape[0]
    xdim = raw_image.shape[1]
    ydim = raw_image.shape[2]
    
    # create filefolds for output
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    if CSV_PATH:
        csv_path = os.path.join(CSV_PATH,'meta_cell')
        img_path = os.path.join(CSV_PATH,'images')
        pathlib.Path(csv_path).mkdir(parents=True, exist_ok=True) 
        pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
    
    # preprocess
    process_image = np.empty(raw_image.shape[:3]+(2,))
    process_image[...,0] = raw_image[...,0]
    process_image[...,1] = raw_image[...,0]
    
    # predict
    app = MultiplexSegmentation()
    labeled_img = app.predict(image=process_image, image_mpp=image_mpp, compartment='nuclear', 
                     postprocess_kwargs_nuclear={'maxima_threshold': 0.6, 'maxima_model_smooth': 0,
                                                   'interior_model_smooth': 2, 'interior_threshold': 0.23,
                                                   'small_objects_threshold': 15,
                                                   'fill_holes_threshold': 15,
                                                   'radius': 3,
                                                   'pixel_expansion': 0})
    
    
    # Save thems to output filefold
    for i in range(len(good_file)):
        save_path = os.path.join(OUTPUT_PATH, good_file[i])
        io.imsave(save_path, labeled_img[i,...,0])
    
    # Save gif
    if gif_fps:
        with imageio.get_writer(uri=os.path.join(OUTPUT_PATH, 'mask.gif'), mode='I', fps=gif_fps) as writer:
            for i in range(len(good_file)):
                save_path = os.path.join(OUTPUT_PATH, good_file[i])
                writer.append_data(imageio.imread(save_path))
    
    # cell track
    if CSV_PATH != None:
    
        if opening_morphology_kwargs != None:
            tracker = CellTracker(opening_morphology_kwargs,'otsu')
        else:
            tracker = CellTracker('otsu')
    tracked_image, info = tracker.track(labeled_img[...,0])
    x, y, area, all_label = zip(*info)
    
    # save tracked images:
    for i in range(tracked_image.shape[0]):
        save_path = os.path.join(img_path, str(i+1)+'.tif')
        io.imsave(save_path, tracked_image[i,...])
        
    # save cell_meta csv
    columns_index = ['cell_id', 'x', 'y', 'pixel_number', 'actual_area']
    for i in range(len(x)):
        df = pd.DataFrame(columns=columns_index)
        frame_label = all_label[i]
        frame_x = x[i]
        frame_y = y[i]
        frame_area = area[i]
        df['cell_id'] = frame_label
        df['x'] = frame_x
        df['y'] = frame_y
        df['pixel_number'] = frame_area
        df['actual_area'] = [a*image_mpp*image_mpp for a in frame_area]
        df.sort_values(by='cell_id')
        path = os.path.join(csv_path, 'meta_cell%d.csv'%(i+1))
        df.to_csv(path, sep=',', header=True, index=True)
