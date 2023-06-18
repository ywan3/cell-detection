# DeGraffenried Microtubules

Our goal is to detect and annotate microtubules that form the structure of the parasite Trypanosoma brucei, and then track
their appearance and disappearance across cross-sections of the parasite.

We use [stardist](https://github.com/stardist/stardist) to try the image segmentation.

For running on CoLab, this may be more helpful: https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_2D_ZeroCostDL4Mic.ipynb#scrollTo=3L9zSGtORKYI

There is also a directory for [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki) Colab notebooks for trying other methods.

# Files

The current main stardist notebook is in `scripts/stardist-test.ipynb`. Data is stored on Oscar, but is also available in a Dropbox folder (ask August for a share).

# Todo and questions:

 * Images in CleanInitial on the Dropbox folder have corresponding annotated and unannotated images, however the images need to be aligned. [Tutorial here](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/) may be helpful.
 * Are there other ways we can preprocess the images? Take crops of just the cell?
 * Wait on Laura for images that have actually **all** of the microtubules annotated, not just in the main image crop.
 * Would incorporating images without MtQ be helpful? Right now there is only the MtQ images in `CleanInitial`.
 * Try a simpler CNN, see if that helps since we are not trying to detect the rest of the objects
 * Play around with parameters in stardist:
  * Drop Alpha channel
  * Decrease learning rate
  * Make image smaller
  * Set number of classes at 5
  * Why is the number of output channels 33 right now?
  * Augment images
  * Try different epochs
  * Try different grid parameters, unet parameters

# CBC Project Information

```
title:
tags:
analysts:
git_repo_url:
resources_used:
summary:
project_id:
```
