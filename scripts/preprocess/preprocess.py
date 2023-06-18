import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import io, transform, feature, draw
from image_align import align_images
from image_crop import random_crop_images
from image_jitter import random_jitter_images








# Perform alignment, random cropping and jittering on tiff images
if __name__ == "__main__":
    
    seg_mask_dir_name = "seg_mask"
    if not os.path.exists(seg_mask_dir_name):
        os.mkdir(seg_mask_dir_name)

    image_path = "./project_data/"
    align_images(image_path)
    # random_crop_images(image_path)
    # random_jitter_images(image_path)

    training_data_path_list = sorted(glob.glob('./project_data/Annotated TIFFs' + '/**/*.tiff', recursive=True))
    cell_images = []
    cell_boundaries = []
    
    for i in range(len(training_data_path_list)):
        
        print(f"Generating segmentation mask for image {i}")
        training_data_path = training_data_path_list[i]
        image_index = re.findall(r'\d+', training_data_path)[0]

        # Load image
        img = Image.open(training_data_path)
        img_arr = np.array(img)
        
        raw_boundary_mask = (img_arr[:,:,2] == 255) & (img_arr[:,:,1] == 0) & (img_arr[:,:,0] == 0)
        
        lines = transform.probabilistic_hough_line(raw_boundary_mask, threshold=0, line_length=80, line_gap=300)

        output_mask = np.zeros_like(raw_boundary_mask)

        for line in lines:
            p0, p1 = line
            rr, cc = draw.line(p0[0], p0[1], p1[0], p1[1])
            output_mask[cc, rr] = 1

        

        dilated_mask = ndimage.binary_dilation(output_mask, structure = np.ones((20, 20))).astype(int)
        # dilated_mask = ndimage.binary_dilation(boundary_mask, structure = struct).astype(int)
        
        

        closed_mask = ndimage.binary_closing(dilated_mask, structure = np.ones((20, 20))).astype(int)
        # closed_mask = ndimage.binary_closing(dilated_mask, structure = struct).astype(int)

        eroded_mask = ndimage.binary_erosion(closed_mask, structure=np.ones((10, 10))).astype(int)

        # Fill in the holes using the new structuring element
        filled_mask = ndimage.binary_fill_holes(eroded_mask, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool))

        # Flip vertically
        flipped_mask = np.flipud(np.rot90(filled_mask, k=1))
        

        # # Show the result
        # plt.imshow(filled_mask, cmap = "gray")
        # plt.show()
        

        # save the segmentation mask as a grayscale TIFF image
        io.imsave(f'./seg_mask/mask_{image_index}.tiff', filled_mask.astype('uint8')*255)
    
    