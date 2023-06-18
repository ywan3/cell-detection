import numpy as np
import cv2
import glob
from stardist import models
from stardist import data
from stardist import matching
import keras
import sklearn
from sklearn.model_selection import train_test_split



training_X_dir_list = ["./project_data/Annotated TIFFs"]
training_Y_dir_list = ["./seg_mask"]

testing_dir_list = [
    "./project_data/New files/With MtQ/Unannotated/TIFFs",
    "./project_data/New files/With MtQ/Without MtQ/Unnanotated/TIFFs",
    "./project_data/CleanInitial/Unannotated"
]

training_X_path_list = []
training_Y_path_list = []
testing_path_list = []

X = []
Y = []
test_data = []

for training_dir in training_X_dir_list:
    training_X_path_list.extend(glob.glob(training_dir + '/**/*.tiff', recursive=True))


for training_dir in training_Y_dir_list:
    training_Y_path_list.extend(glob.glob(training_dir + '/**/*.tiff', recursive=True))

for testing_dir in testing_dir_list:
    testing_path_list.extend(glob.glob(training_dir + '/**/*.tiff', recursive=True))

for training_path in training_X_path_list:
    X.append(cv2.imread(training_path, cv2.IMREAD_GRAYSCALE))

for training_path in training_Y_path_list:
    Y.append(cv2.imread(training_path, cv2.IMREAD_GRAYSCALE))

for testing_path in testing_path_list:
    test_data.append(cv2.imread(testing_path, cv2.IMREAD_GRAYSCALE))



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


model = models.StarDist2D(
    name='stardist'
)

callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print("Epoch:", epoch))


print(X_train[0].shape)
print(Y_train[0].shape)


# Train model
model.train(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50)

mean_average_precision = matching.mean_average_precision(Y_test, model.predict(X_test))

# precision recall needed
# precision thresholding => 

print(f"Mean Average Precision: {mean_average_precision}")