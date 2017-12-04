'''

USE : BINARY CLASSIFICATION TO FIND DECIMAL VALUE FROM IMAGE
Created by : Mayur Kanojiya


prerequisite : need pre-trained model by keras with following model config

{"config": [{"config": {"kernel_size": [3, 3], "filters": 32, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_1", "activation": "relu", "bias_regularizer": null, "dtype": "float32", "trainable": true, "use_bias": true, "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "padding": "same", "bias_constraint": null, "activity_regularizer": null, "kernel_constraint": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1], "batch_input_shape": [null, 3, 28, 28]}, "class_name": "Conv2D"}, {"config": {"kernel_size": [3, 3], "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_2", "activation": "relu", "trainable": true, "kernel_constraint": null, "use_bias": true, "filters": 32, "padding": "valid", "bias_constraint": null, "activity_regularizer": null, "bias_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1]}, "class_name": "Conv2D"}, {"config": {"padding": "valid", "name": "max_pooling2d_1", "pool_size": [2, 2], "data_format": "channels_first", "strides": [2, 2], "trainable": true}, "class_name": "MaxPooling2D"}, {"config": {"name": "dropout_1", "noise_shape": null, "seed": null, "rate": 0.2, "trainable": true}, "class_name": "Dropout"}, {"config": {"kernel_size": [3, 3], "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_3", "activation": "relu", "trainable": true, "kernel_constraint": null, "use_bias": true, "filters": 64, "padding": "same", "bias_constraint": null, "activity_regularizer": null, "bias_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1]}, "class_name": "Conv2D"}, {"config": {"kernel_size": [3, 3], "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_4", "activation": "relu", "trainable": true, "kernel_constraint": null, "use_bias": true, "filters": 64, "padding": "valid", "bias_constraint": null, "activity_regularizer": null, "bias_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1]}, "class_name": "Conv2D"}, {"config": {"padding": "valid", "name": "max_pooling2d_2", "pool_size": [2, 2], "data_format": "channels_first", "strides": [2, 2], "trainable": true}, "class_name": "MaxPooling2D"}, {"config": {"name": "dropout_2", "noise_shape": null, "seed": null, "rate": 0.2, "trainable": true}, "class_name": "Dropout"}, {"config": {"kernel_size": [3, 3], "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_5", "activation": "relu", "trainable": true, "kernel_constraint": null, "use_bias": true, "filters": 128, "padding": "same", "bias_constraint": null, "activity_regularizer": null, "bias_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1]}, "class_name": "Conv2D"}, {"config": {"kernel_size": [3, 3], "kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "dilation_rate": [1, 1], "kernel_regularizer": null, "data_format": "channels_first", "name": "conv2d_6", "activation": "relu", "trainable": true, "kernel_constraint": null, "use_bias": true, "filters": 128, "padding": "valid", "bias_constraint": null, "activity_regularizer": null, "bias_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "strides": [1, 1]}, "class_name": "Conv2D"}, {"config": {"padding": "valid", "name": "max_pooling2d_3", "pool_size": [2, 2], "data_format": "channels_first", "strides": [2, 2], "trainable": true}, "class_name": "MaxPooling2D"}, {"config": {"name": "dropout_3", "noise_shape": null, "seed": null, "rate": 0.2, "trainable": true}, "class_name": "Dropout"}, {"config": {"name": "flatten_1", "trainable": true}, "class_name": "Flatten"}, {"config": {"kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "name": "dense_1", "kernel_regularizer": null, "activation": "relu", "bias_regularizer": null, "trainable": true, "use_bias": true, "bias_constraint": null, "activity_regularizer": null, "kernel_constraint": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "units": 512}, "class_name": "Dense"}, {"config": {"kernel_initializer": {"config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}, "class_name": "VarianceScaling"}, "name": "dense_2", "kernel_regularizer": null, "activation": "softmax", "bias_regularizer": null, "trainable": true, "use_bias": true, "bias_constraint": null, "activity_regularizer": null, "kernel_constraint": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "units": 2}, "class_name": "Dense"}], "keras_version": "2.0.9", "class_name": "Sequential", "backend": "tensorflow"}

input_size is used 28 for training so 28 sized image will go in convolutional model


'''

from keras.models import load_model
import cv2
import numpy as np
from skimage import transform

IMG_SIZE = 28
model = load_model('model_decimal.h5')


#function to preprocess the image
def preprocess_img(img):
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')
    img = np.rollaxis(img, -1)

    return img



#function to find decimal exist in frame or not
def isDecimal(image):
    image = preprocess_img(image)
    image = np.array([image], dtype='float32')

    y_pred = model.predict_classes(image)
    if(y_pred[0] == 1):
        return True
    else:
        return False


image = cv2.imread('Dataset/training-images/nondecimal/967.png')

print(isDecimal(image))
