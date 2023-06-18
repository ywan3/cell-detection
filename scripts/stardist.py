from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os.path
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, normalize

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

Y = sorted(glob('/gpfs/data/cbc/cdegraffenried/degraffenried_microtubules/data/test/annotated/*.tiff'))[32:70]
#print(list(map(os.path.basename, Y)))
#print("length of Y: ", len(Y))

X = sorted(glob('/gpfs/data/cbc/cdegraffenried/degraffenried_microtubules/data/test/unannotated/*.tiff'))[34:71]
#print(list(map(os.path.basename, X)))
#print("length of X: ", len(X))

assert(len(X)==len(Y))
assert(list(map(os.path.basename,Y))==list(map(os.path.basename,X)))

Y = list(map(imread,Y))
X = list(map(imread,X))

Y_new = [np.select([(img[:,:,0]<55) & (img[:,:,1]<55) & (img[:,:,2]>200), # blue for microtubules
                    (img[:,:,0]<55) & (img[:,:,1]>200) & (img[:,:,2]<55), # green for MtQ region
                    (img[:,:,0]>200) & (img[:,:,1]>200) & (img[:,:,2]<55), # yellow for MtQ region
                    (img[:,:,0]>200) & (img[:,:,1]<55) & (img[:,:,2]>200), # pink for MtQ region
                    (img[:,:,0]<55) & (img[:,:,1]>200) & (img[:,:,2]>200) # cyan for MtQ region
               ], [1,2,3,4,5], 0) for img in Y]

img, lbl = X[0], fill_label_holes(Y_new[0])
assert img.ndim in (2,3)
img = img if img.ndim==2 else img[...,:3] # this is copied from stardist jupyter notebook but don't know what it means
# assumed axes ordering of img and lbl is: YX(C)

n_rays = [2**i for i in range(2,8)]
scores = []
for r in tqdm(n_rays):
    Y_reconstructed = [relabel_image_stardist(lbl, n_rays=r) for lbl in Y_new]
    mean_iou = matching_dataset(Y_new, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
    scores.append(mean_iou)

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

#X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y_new)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
)
print(conf)
vars(conf)

model = StarDist2D(conf, name='stardist', basedir='models')

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

model.train(X_trn, Y_trn, validation_data=(X_val,Y_val))

model.optimize_thresholds(X_val, Y_val)

Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

stats[taus.index(0.5)]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
    ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax1.set_xlabel(r'IoU threshold $\tau$')
ax1.set_ylabel('Metric value')
ax1.grid()
ax1.legend()

for m in ('fp', 'tp', 'fn'):
    ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax2.set_xlabel(r'IoU threshold $\tau$')
ax2.set_ylabel('Number #')
ax2.grid()
ax2.legend();
