#!/usr/bin/env python3
# -*- coding: utf-8 -*-# USAGE
# python train.py

# import the necessary packages
from model.data_utils import process_input
from model import config
from model import subpixel_net
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf

def psnr(orig, pred):
	# cast the target images to integer
	orig = orig * 255.0
	orig = tf.cast(orig, tf.uint8)
	orig = tf.clip_by_value(orig, 0, 255)
  
	# cast the predicted images to integer
	pred = pred * 255.0
	pred = tf.cast(pred, tf.uint8)
	pred = tf.clip_by_value(pred, 0, 255)
  
	# return the psnr
	return tf.image.psnr(orig, pred, max_val=255)

# define autotune flag for performance optimization
AUTO = tf.data.AUTOTUNE

# load the image paths from disk and initialize TensorFlow Dataset
# objects
print("[INFO] loading images from disk...")
trainPaths = list(paths.list_images(config.TRAIN_SET))
valPaths = list(paths.list_images(config.VAL_SET))
trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
valDS = tf.data.Dataset.from_tensor_slices(valPaths)

# prepare data loaders
print("[INFO] preparing data loaders...")
trainDS = trainDS.map(process_input,
					  num_parallel_calls=AUTO).batch(
	config.BATCH_SIZE).prefetch(AUTO)
valDS = valDS.map(process_input,
				  num_parallel_calls=AUTO).batch(
	config.BATCH_SIZE).prefetch(AUTO)

# initialize, compile, and train the model
print("[INFO] initializing and training model...")
model = subpixel_net.get_subpixel_net()
model.compile(optimizer="adam", loss="mse", metrics=psnr)
H = model.fit(trainDS, validation_data=valDS, epochs=config.EPOCHS)

# create output directory if not exists
if not os.path.exists(config.OUT_DIR):
    print("[INFO] creating output directory...")
    os.makedirs(config.OUT_DIR)
    
# prepare training plot of the model and serialize it
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["psnr"], label="train_psnr")
plt.plot(H.history["val_psnr"], label="val_psnr")
plt.title("Training Loss and PSNR")
plt.xlabel("Epoch #")
plt.ylabel("Loss/PSNR")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT)

# serialize the trained model
print("[INFO] serializing model...")
model.save(config.SUPER_RES_MODEL)
