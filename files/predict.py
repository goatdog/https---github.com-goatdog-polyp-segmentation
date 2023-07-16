import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from keras.utils import CustomObjectScope
from data import load_data, tf_dataset
from train import check

def read_image2(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask2(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    path = "dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    batch_size = 8
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    test_steps = len(test_x) // batch_size
    if len(test_x) % batch_size != 0: test_steps += 1
    with CustomObjectScope({'check': check}):
        model = tf.keras.models.load_model("files/model.h5")
        
    model.evaluate(test_dataset, steps=test_steps)
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image2(x)
        y = read_mask2(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))
        y_pred = y_pred[0] > 0.5
        h, w, a = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        all_images = [
            x * 255.0, white_line, 
            mask_parse(y), white_line, 
            mask_parse(y_pred) * 255.0
        ]
        
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)
        
