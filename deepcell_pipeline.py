"""
This is a DeepCell python script which enables you to segment images with spacial or chronological order.

Please note that:

1. All the images must have same size.

2. RGB or RGBA images are not supported, so your images have one channel only!

3. The filename of images should have number suffix, which shows their sequences.

    For example:
    images name: "roundh11_A1_1.tif","roundh11_A1_2.tif","roundh11_A1_3.tif"
    instead of "roundh11_A1_1_z.tif","roundh11_A1_2_z.tif","roundh11_A1_3_z.tif";"roundh11_A1_995500002.tif","roundh11_A1_454600003.tif".

so that the code can know your image sequence.

4. The filefold can contain files that are NOT images, but if there are other images in your input filefold with different format from your input image, use '-f format' to appoint the processed image format. 
    For example, if you use '-f png', Deepcell will try to segment all your images with '.png' suffix only. 
"""


### VERSION = 2.0.0 ###
# 1. Added Cell-Track
# 2. CSV
# Get parameters from command line



import argparse as arg

parser = arg.ArgumentParser()

parser.add_argument("-i", "--imput", type=str, nargs=1, required=True,
                    help="[Necessary]Input filefold. The absolute path to your filefold which contains your raw images or relative path if it is in the same filefold as this python script. It must end with a '/'(in linux).")

#group.add_argument("-n", "--name", type=str, required=False, action="append", nargs='+',
#                  help="[Necessary]Input images. Instead of using '-i ./filefold_name', you can use '-n ./filefold_name/***1.tif ./filefold_name/***2.tif' to segment the exact images you wish to process. Moreover, you can append this parameter, such as '-n 1.png 2. png -m 0.65 -n 3.img'. Note that you cannot use '-n' and  '-i' at the same time.")

parser.add_argument("-o", "--output", type=str, nargs=1, required=True,
                   help="[Necessary]Output filefold. The absolute path to filefold which saves processed images or relative path if it is in the same filefold as this python script. If the filefold does not exist, it will create one. It must end with a '/'(in linux).")

parser.add_argument("-m", "--mpp", type=float, nargs=1, required=True, default=0.1,
                   help="[Necessary]mpp: microns per pixon, the physical distance of a pixon. Default:0.1(for MERFISH images).It is key parameter of Deepcell model. If it performs bad on your cell images, it is very likely that the value of mpp is inappropriate. So we recommend that you can try some value to test on your images.")

parser.add_argument("-z", "--zdistance", type=float, nargs=1, required=False, 
                    help="[Optional]It is the physical distance between two z-stacks, and the unit is micron. There is no need adding this parameter if your images are formed in chronological order.")

parser.add_argument("-c", "--csv", type=str, nargs=1, required=False,
                   help="[Necessary]Output csv file. The absolute path to save the .csv file which contains property of cells in your images or relative path if it is in the same filefold as this python script. ")

parser.add_argument("-f", "--format", type=str, nargs='+', required=False,
                    help="[Optional]Default:None. You can add one or more image format name after '-f', such as '-f png tif' and '-f tiff'. This will tell Deepcell to predict the images with the format you specified only in your input filefold path. If this paramater is not given, then Deepcell will try to predict all your images in your filefold. ")

parser.add_argument("-g", "--gif", required=False, type=int, nargs=1,
                    help="[Optional]Add this parameter, DeepCell will create a gif file of spatially or chronologically sorted masks, which will be saved in your output filefold. You need to use '-g 6' to set fps to 6. Input bigger number to make gif play faster.")


parser.add_argument("-k", "--ksize", required=False, type=tuple, nargs=1,
                    help="[Optional]Kernal size of eroding and diluting. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

parser.add_argument("-e", "--ero", required=False, type=int, nargs=1,
                    help="[Optional]Iteration times of eroding. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

parser.add_argument("-d", "--dil", required=False, type=int, nargs=1,
                    help="[Optional]Iteration times of diluting. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

args = parser.parse_args()
INPUT_PATH = args.imput[0]
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
    

# File preprocess:
import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import suffix, SortFilename

all_file = os.listdir(INPUT_PATH)
all_file = SortFilename(all_file)
target_file = []
good_file = []
if format_list:
    for file in all_file:
        if suffix(file) in format_list:
            target_file.append(file)
        else:
            continue
else:
    target_file = all_file

# Decide whether to create output filefold
import pathlib

pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
if CSV_PATH:
    pathlib.Path(CSV_PATH).mkdir(parents=True, exist_ok=True) 
    
# Load images

raw_image = []
for file in target_file:
    abs_path = INPUT_PATH + file
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

# Get batch size and image size
batch_size = raw_image.shape[0]
xdim = raw_image.shape[1]
ydim = raw_image.shape[2]

process_image = np.empty(raw_image.shape[:3]+(2,))
process_image[...,0] = raw_image[...,0]
process_image[...,1] = raw_image[...,0]

# Load model
from deepcell.applications.mesmer import Mesmer as MultiplexSegmentation

app = MultiplexSegmentation()

# predict
labeled_img = app.predict(image=process_image, image_mpp=image_mpp, compartment='nuclear', 
                     
                     postprocess_kwargs_nuclear={'maxima_threshold': 0.6, 'maxima_model_smooth': 0,
                                                   'interior_model_smooth': 2, 'interior_threshold': 0.23,
                                                   'small_objects_threshold': 15,
                                                   'fill_holes_threshold': 15,
                                                   'radius': 3,
                                                   'pixel_expansion': 0})

# add overlay
#from deepcell.utils.plot_utils import make_outline_overlay

#mask = make_outline_overlay(labeled_img, labeled_img)

import imageio
from skimage import io

# Save thems to output filefold
for i in range(len(good_file)):
    save_path = OUTPUT_PATH + good_file[i]
    io.imsave(save_path, mask[i,...,0])
    
# Save gif
if gif_fps:
    with imageio.get_writer(uri=OUTPUT_PATH+'mask.gif', mode='I', fps=gif_fps) as writer:
        for i in range(len(good_file)):
            save_path = OUTPUT_PATH + good_file[i]
            writer.append_data(imageio.imread(save_path))


# CellTracker
from celltracker import CellTracker

if CSV_PATH != None:
    
    background = np.full(labeled_img.shape, 255)
    mask = make_outline_overlay(background, labeled_img)
    
    if opening_morphology_kwargs != None:
        tracker = CellTracker(opening_morphology_kwargs=opening_morphology_kwargs)
    else:
        tracker = CellTracker()
    
    tracked_image, info = tracker.track(mask)
    x, y, area, all_label = zip(*info)
    
    tracked_image_save_path = CSV_PATH+'/tracked_images'
    pathlib.Path(tracked_image_save_path).mkdir(parents=True, exist_ok=True)
    for i in range(tracked_image.shape[0]):
        plt.figure(dpi=400)
        plt.imshow(tracked_image[i],cmap="gray")
        frame_x = x[i]
        frame_y = y[i]
        for j in range(1,len(frame_x)):
            plt.text(frame_x[j], frame_y[j],s=str(all_label[i][j]),color="yellow",fontsize=6)
        plt.savefig(tracked_image_save_path+"_%d.png"%(i+1))
        plt.close()
    # save to gif
    with imageio.get_writer(uri='tracked_images.gif', mode='I', fps=5) as writer:
        for i in range(len(tracked_image)):
            writer.append_data(imageio.imread(tracked_image_save_path+"_%d.png"%(i+1)))
    
    # save to csv
    # one frame, one csv
    #   cell id    centroid x   centroid y   number of pixels   actual area
    #      1           25           58            ***               ****
    #      *            *            *              *                 *
    columns_index = ['cell_id', 'x', 'y', 'pixel_number', 'actual_area']
    for i in range(len(x)):
        df = pd.DataFrame(columns=columns_index)
        frame_label = all_label[i]
        frame_x = x[i]
        frame_y = y[i]
        frame_area = area[i]
        df['cell_id'] = frame_label
        df['x'] = frame_x
        df['y'] = frame_
        df['pixel_number'] = frame_area_1
        df['actual_area'] = [a*image_mpp*image_mpp for a in frame_area]
        df.sort_values(by='cell_id')
        df.drop(index=0)
        df.to_csv(CSV_PATH + '/' +'celltracked_frame%d'%(i+1)+'.csv', sep=',', header=True, index=True)