from keras.models import load_model
import cv2
import numpy as np
from skimage import transform
IMG_SIZE = 28


#function to preprocess the image
def preprocess_img(img):

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')
    img = np.rollaxis(img, -1)

    return img



model = load_model('model_decimal.h5')




import os
for image in os.listdir('Dataset/testing-images/decimal/'):

    imgs1 = []
    image = cv2.imread(os.path.join('Dataset/testing-images/decimal/',image))
    image = preprocess_img(image)
    imgs1.append(image)
    X1 = np.array(imgs1, dtype='float32')

    y_pred = model.predict_classes(X1)

    print(y_pred)
    # print(model.predict_classes(img))
    print(model.predict_proba(X1))

    cv2.waitKey(200)
    input("enter")
