import cv2
import numpy as np
import os
import glob
from util import crop_image_list, get_folder_structure, save_image




# Global variables

#  output directory for aligned images
output_dir = './cropped_images/'

# input directory for hierarchy mapping
from_dir = './project_data/'


# set the dimensions of the cropped images
crop_width = 256
crop_height = 256

# set the number of random crops to generate per image
num_crops = 5


# Augmentation Method II:
def random_crop_images(file_path):
    
    print("----- Creating Directory for cropped images -----")
    
    os.makedirs(output_dir, exist_ok=True)

    # Mimic the original data structure in folder, ignoring the empty string
    folder_path_list = get_folder_structure(file_path)[1:]

    for folder_path in folder_path_list:
        
        # Create corresponding folder structure
        new_path = folder_path.split('/', 2)[2]
        os.makedirs(output_dir + new_path, exist_ok=True)

        image_path_list = glob.glob(folder_path + "/*.tiff")
        if image_path_list == []:
            continue
        crop_image_list(image_path_list, folder_path, output_dir, crop_height, crop_width, num_crops)

