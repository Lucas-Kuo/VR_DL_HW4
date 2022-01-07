#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# USAGE
# python generate_super_res.py
# import the necessary packages
from model import config
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from cv2 import imwrite
import os
 
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

def load_image(imagePath):
	# load image from disk and downsample it using the bicubic method
	orig = load_img(imagePath)
	downsampled = orig.resize((orig.size[0] // config.DOWN_FACTOR,
		orig.size[1] // config.DOWN_FACTOR), Image.BICUBIC)
    
	# return a tuple of the original and downsampled image
	return (orig, downsampled)

def get_y_channel(image):
	# convert the image to YCbCr colorspace and then split it to get the
	# individual channels
	ycbcr = image.convert("YCbCr")
	(y, cb, cr) = ycbcr.split()
    
	# convert the y-channel to a numpy array, cast it to float, and
	# scale its pixel range to [0, 1]
	y = np.array(y)
	y = y.astype("float32") / 255.0
    
	# return a tuple of the individual channels
	return (y, cb, cr)

def clip_numpy(image):
	# cast image to integer, clip its pixel range to [0, 255]
	image = tf.cast(image * 255.0, tf.uint8)
	image = tf.clip_by_value(image, 0, 255).numpy()
    
	# return the image
	return image

def postprocess_image(y, cb, cr):
	# do a bit of initial preprocessing, reshape it to match original
	# size, and then convert it to a PIL Image
	y = clip_numpy(y).squeeze()
	y = y.reshape(y.shape[0], y.shape[1])
	y = Image.fromarray(y, mode="L")
	
	# resize the other channels of the image to match the original
	# dimension
	outputCB= cb.resize(y.size, Image.BICUBIC)
	outputCR= cr.resize(y.size, Image.BICUBIC)
	
	# merge the resized channels altogether and return it as a numpy
	# array
	final = Image.merge("YCbCr", (y, outputCB, outputCR)).convert("RGB")
	return np.array(final)

# load the test image paths from disk
print("[INFO] loading test images...")
testPaths = list(paths.list_images(config.TEST_SET))

# load our super-resolution model from disk
print("[INFO] loading model...")
superResModel = load_model(config.SUPER_RES_MODEL,
	custom_objects={"psnr" : psnr})

# iterate through our test image paths
print("[INFO] performing predictions...")
for (i, path) in enumerate(testPaths):
	# grab the original and the downsampled images from the
	# current path
	(orig, downsampled) = load_image(path)
	
	# retrieve the individual channels of the current image and perform
	# inference
	(y, cb, cr) = get_y_channel(orig)
	upscaledY = superResModel.predict(y[None, ...])[0]
	
	# postprocess the output and apply the naive bicubic resizing to
	# the downsampled image for comparison
	finalOutput = postprocess_image(upscaledY, cb, cr)
	# naiveResizing = downsampled.resize(orig.size, Image.BICUBIC)
	
	# save the answer
	save_path =  os.path.join(config.ANSWER_PATH, f"{path[:-4]}_pred.png") # trim off the .xxx of path
	predict_img = Image.fromarray(finalOutput)
	predict_img.save(save_path)
	
	# visualize the results and save them to disk
	path = os.path.join(config.VISUALIZATION_PATH, f"{path[:-4]}_viz.png")
	(fig, (ax1, ax2)) = plt.subplots(ncols=2, figsize=(12, 12))
	ax1.imshow(naiveResizing)
	ax2.imshow(finalOutput.astype("int"))
	ax1.set_title("Original")
	ax2.set_title("Super-res Model")
	fig.savefig(path, dpi=300, bbox_inches="tight")
