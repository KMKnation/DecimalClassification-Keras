'''

USE : TO EVALUATE BINARY CLASSIFICATION MODEL
Created by : Mayur Kanojiya

'''

import h5py
import numpy as np
from skimage import io, color, exposure, transform
import glob, os
from keras.models import Sequential, model_from_json
import cv2

NUM_CLASSES = 2
IMG_SIZE = 28


from  get_set_class_index import getAllIndexFromCharacter
def get_class(img_path):
    return int(getAllIndexFromCharacter(img_path.split('/')[-2]))

from get_set_class_index import getAllCharacterFromIndex
def get_class_name(index):
    return int(getAllCharacterFromIndex(index))

#function to preprocess the image
def preprocess_img(img):

    try:
        # histogram normalization in y
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
    except (ValueError):
        img = np.stack((img,) * 3)
        img = np.transpose(img, (1, 2, 0))
        pass


    # central crop
    # min_side = min(img.shape[:-1])
    # centre = img.shape[0]//2, img.shape[1]//2
    # img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
    #       centre[1] - min_side // 2:centre[1] + min_side // 2,:]

    #rescale to standard size
    # img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # cv2.imshow('img', img)
    # cv2.waitKey()

    #roll color axis to axis 0
    img = np.rollaxis(img, -1)


    return img


'''Load training images into numpy array'''
try:
    with h5py.File('XDemo.h5', 'r') as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]

        print('Length of X is {} and Length Of Y is {}'.format(len(X), len(Y)))
        print("Loaded images from X_test.h5")
except (IOError, OSError):
    print("Error occured while reading X_test.h5")

    root_dir = 'Dataset/testing-images'

    imgs = []
    labels = []

    all_img_paths = glob.glob((os.path.join(root_dir, '*/*.png')))
    np.random.shuffle(all_img_paths)

    for img_path in all_img_paths:
        try:
            import cv2
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if (len(imgs) % 1000 == 0): print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    print('Length of X is {} and Length Of Y is {}'.format(len(X), len(Y)))
    with h5py.File('X_test.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

# model = Sequential()
#load json model

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model.h5')
from keras.models import load_model
model = load_model('model_decimal.h5')


y_pred = model.predict_classes(X)

print(y_pred)

#converting into eye because we got classes index and our label is converted into matrix
y_pred = np.eye(NUM_CLASSES, dtype='uint8')[y_pred]

acc = np.sum(y_pred==Y)/np.size(y_pred)
print("Test accuracy = {}".format(acc))


