import cv2
import numpy as np
import os
import glob
from util import align_image_list, get_folder_structure, save_image


# Global variables

#  output directory for aligned images
output_dir = './aligned_images/'

# input directory for hierarchy mapping
from_dir = './project_data/'

# reference image_path
ref_path = "./project_data/Annotated TIFFs/zap000.tiff"



# Augmentation Method I:
def align_images(file_path):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # Mimic the original data structure in folder, ignoring the empty string
    folder_path_list = get_folder_structure(file_path)
    
    for folder_path in folder_path_list:
        print(folder_path)
        # Create corresponding folder structure
        new_path = folder_path.split('/', 2)[2]
        os.makedirs(output_dir + new_path, exist_ok=True)

        
        image_path_list = glob.glob(folder_path + "/*.tiff")
        if image_path_list == []:
            continue

        align_image_list(image_path_list, folder_path, output_dir, ref_image=ref_image)


