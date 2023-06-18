from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob
from stardist import models
from stardist import data
from stardist import matching
import sys


testing_dir_list = [
    "./project_data/New files/With MtQ/Unannotated/TIFFs",
    "./project_data/New files/With MtQ/Without MtQ/Unnanotated/TIFFs",
    "./project_data/CleanInitial/Unannotated"
]


# prints a list of available models
models.StarDist2D.from_pretrained()

# creates a pretrained model
model = models.StarDist2D.from_pretrained('2D_versatile_fluo')

num_images = 4
num_rows = num_images
num_cols = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))


os.makedirs("./pretrained_labels", exist_ok = True)

i = 0
imgs = []
for directory in testing_dir_list:
    for file in glob.glob(directory + '/*.tiff'):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        labels, _ = model.predict_instances(normalize(img), nms_thresh=0.9)

        # Display the input image
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Input image")

        # Display the predicted labels
        axes[i, 1].imshow(render_label(labels, img=img))
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Prediction + input overlay")
        
        filename = f"{file}_labels.tiff"
        cv2.imwrite(filename, labels)

        i = i + 1
        if i == 3:
            # Adjust spacing between subplots
            plt.subplots_adjust(hspace=0.5)

            # Display the plot
            plt.show()
            
        

