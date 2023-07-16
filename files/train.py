import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from model import build_model

def precheck(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    x = x.astype(np.float32)
    return x

def check(y_true, y_pred):
    return tf.numpy_function(precheck, [y_true, y_pred], tf.float32)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    path = "dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    # print(len(train_x), len(valid_x), len(test_x))
    batch_size = 8
    lr = 1e-4
    epoch = 20
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    model = build_model()
    opt = tf.keras.optimizers.Adam(lr)
    metric = ["acc", Recall(), Precision(), check]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metric)
    callback = [
        ModelCheckpoint("files/model.h5"), 
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),  
        CSVLogger("files/data.csv"), 
        TensorBoard(), 
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
    ]
    
    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size
    if len(train_x) % batch_size != 0: train_steps += 1
    if len(valid_x) % batch_size != 0: valid_steps += 1
    
    model.fit(
        train_dataset, 
        validation_data=valid_dataset,
        epochs=epoch, 
        steps_per_epoch=train_steps, 
        validation_steps=valid_steps, 
        callbacks=callback, 
        shuffle=False
    )