import cv2
import numpy as np
import os


# processes a bottom-level folder that contains only tiff images
def align_image_list(image_path_list, folder_path, output_dir, ref_image=None):
    
    #print("----- Loading Images for Aligning-----")
    # Read in the list of TIFF images
    images = []
    
    for f in image_path_list:
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        images.append(image)
    
    #print("----- Setting Reference Image -----")
    # Define the reference image and rescale intensity
    if ref_image is None:
        ref_image = images[0]

    # Define the registration method
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 1000
    termination_eps = 0.01
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Loop over the remaining images and align them to the reference image
    for i in range(1, len(images)):
        #print("Aligning Image Number: {}".format(i))
        moving_image = images[i]
        
        try:
            # Use OpenCV's registration method to align the images
            (cc, warp_matrix) = cv2.findTransformECC(moving_image, ref_image, warp_matrix, warp_mode, criteria)

            # Apply the transformation to the moving image
            aligned_image = cv2.warpAffine(moving_image, warp_matrix, (moving_image.shape[1], moving_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            new_path = folder_path.split('/', 2)[2]
            
            save_image(aligned_image, new_path, i, output_dir, "aligned")
        except Exception:
            # Dealing with findTransformECC not converging
            continue

        #print("Image Alignment Finished")



def crop_image_list(image_path_list, folder_path, output_dir, crop_height, crop_width, num_crops):
    
    print("----- Loading Images for Cropping-----")
    # Read in the list of TIFF images
    images = []
    
    for f in image_path_list:
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        images.append(image)
    

    for i in range(len(images)):
        print("Cropping Image Number: {}".format(i))
        image = images[i]

        # Get the dimensions of the image
        height, width = image.shape

        # Generate random crop coordinates
        x = np.random.randint(0, width - crop_width)
        y = np.random.randint(0, height - crop_height)

        # Crop the image
        cropped_image = image[y:y + crop_height, x:x + crop_width]

        # Do something with the cropped image (e.g. save it)
        new_path = folder_path.split('/', 2)[2]
            
        save_image(cropped_image, new_path, i, output_dir, "cropped")

    print("Image Cropping Finished")


def jitter_image_list(image_path_list, folder_path, output_dir, angle_range, shift_range):
    
    print("----- Loading Images for Jittering-----")
    # Read in the list of TIFF images
    images = []
    
    for f in image_path_list:
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    for i in range(len(images)):
        print("Jittering Image Number: {}".format(i))
        image = images[i]

        # Randomly rotate the image
        angle = np.random.uniform(-angle_range, angle_range)
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

        # Randomly shift the image
        x_shift, y_shift = np.random.randint(-shift_range, shift_range, size=2)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        jittered_image = cv2.warpAffine(image, M, (cols, rows))
        

        # Do something with the cropped image (e.g. save it)
        new_path = folder_path.split('/', 2)[2]
            
        save_image(jittered_image, new_path, i, output_dir, "jittered")

    print("Image Jittering Finished")



# Get list of all subdirectory names given top-level directory name
def get_folder_structure(dir_path):
    
    bottom_level_folders = []
    
    for root, dirs, files in os.walk(dir_path):
        if not dirs:  # If there are no subdirectories
            bottom_level_folders.append(root)

    return bottom_level_folders



def save_image(aligned_image, image_path, i, output_dir, type_string):
    # Save the aligned image
    image_path = os.path.join(output_dir, image_path, f'{type_string}_{i}.tiff')
    cv2.imwrite(image_path, aligned_image)








# extract training image boundaries
def extract_boundary(training_data):
    pass




# morphological reconstruction to get object segmentation mask
def morph_reconstruction():
    pass


# morphological dilation to get boundary segmentation mask
def morph_dilation():
    pass

# contour determination using findContour from opencv
def find_contour():
    pass


# sort by geometric property like angle to obtain clockwise order in coordinates list
def sort_by_angle():
    pass



