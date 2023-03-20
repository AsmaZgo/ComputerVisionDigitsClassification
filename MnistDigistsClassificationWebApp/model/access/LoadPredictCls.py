import cv2
from keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
import numpy as np
class LoadPredictCls:
    def __int__(self):
        pass

    def set_path_model(self, path):
        self.model_path = path

    def read_image_from_path(self, image_file_path):
        image = cv2.imread(image_file_path, cv2.IMREAD_ANYDEPTH)
        return image

    def preprocess_data(self, image):
        image_size = image.shape[1]
        input_size = image_size * image_size
        image = np.reshape(image, [-1, input_size])
        im_pred = tf.cast(image, tf.float32) / 255.0
        return im_pred

    def load_model_and_predict(self, img):
        loaded_model = keras.models.load_model(self.model_path)
        im_pred_y = loaded_model.predict(img)
        return np.argmax(im_pred_y, axis=1)[0]
