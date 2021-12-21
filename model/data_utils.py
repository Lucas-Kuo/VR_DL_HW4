# import the necessary packages
from . import config
import tensorflow as tf

def process_input(imagePath, downFactor=config.DOWN_FACTOR):
	# determine size of the downsampled images
	resizeShape = config.ORIG_SIZE[0] // downFactor
  
	# load the original image from disk, decode it as a JPEG image,
	# scale its pixel values to [0, 1] range, and resize the image
	origImage = tf.io.read_file(imagePath)
	origImage = tf.image.decode_jpeg(origImage, 3)
	origImage = tf.image.convert_image_dtype(origImage, tf.float32)
	origImage = tf.image.resize(origImage, config.ORIG_SIZE,
		method="area")
  
  # convert the color space from RGB to YUV and only keep the Y
	# channel (which is our target variable)
	origImageYUV = tf.image.rgb_to_yuv(origImage)
	(target, _, _) = tf.split(origImageYUV, 3, axis=-1)
  
	# resize the target to a lower resolution
	downImage = tf.image.resize(target, [resizeShape, resizeShape],
		method="area")
  
	# clip the values of the input and target to [0, 1] range
	target = tf.clip_by_value(target, 0.0, 1.0)
	downImage = tf.clip_by_value(downImage, 0.0, 1.0)
  
	# return a tuple of the downsampled image and original image
	return (downImage, target)
