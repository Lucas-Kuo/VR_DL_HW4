# import the necessary packages
from . import config
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

def rdb_block(inputs, numLayers):
	# determine the number of channels present in the current input
	# and initialize a list with the current inputs for concatenation
	channels = inputs.get_shape()[-1]
	storedOutputs = [inputs]
  
  # iterate through the number of residual dense layers
	for _ in range(numLayers):
		# concatenate the previous outputs and pass it through a
		# CONV layer, and append the output to the ongoing concatenation
		localConcat = tf.concat(storedOutputs, axis=-1)
		out = Conv2D(filters=channels, kernel_size=3, padding="same",
			activation="relu",
			kernel_initializer="Orthogonal")(localConcat)
		storedOutputs.append(out)
    
    # concatenate all the outputs, pass it through a pointwise
	# convolutional layer, and add the outputs to initial inputs
	finalConcat = tf.concat(storedOutputs, axis=-1)
	finalOut = Conv2D(filters=inputs.get_shape()[-1], kernel_size=1,
		padding="same", activation="relu",
		kernel_initializer="Orthogonal")(finalConcat)
	finalOut = Add()([finalOut, inputs])
    
	# return the final output
	return finalOut


def get_subpixel_net(downsampleFactor=config.DOWN_FACTOR, channels=1,
	rdbLayers=config.RDB_LAYERS):
	# initialize an input layer
	inputs = Input((None, None, 1))
    
	# pass the inputs through a CONV => CONV block
	x = Conv2D(64, 5, padding="same", activation="relu",
		kernel_initializer="Orthogonal")(inputs)
	x = Conv2D(64, 3, padding="same", activation="relu",
		kernel_initializer="Orthogonal")(x)
    
	# pass the outputs through an RDB => CONV => RDB block
	x = rdb_block(x, numLayers=rdbLayers)
	x = Conv2D(32, 3, padding="same", activation="relu",
		kernel_initializer="Orthogonal")(x)
	x = rdb_block(x, numLayers=rdbLayers)
    
    # pass the inputs through a final CONV layer such that the
	# channels of the outputs can be spatially organized into
	# the output resolution
	x = Conv2D(channels * (downsampleFactor ** 2), 3, padding="same",
		activation="relu", kernel_initializer="Orthogonal")(x)
	outputs = tf.nn.depth_to_space(x, downsampleFactor)
    
	# construct the final model and return it
	model = Model(inputs, outputs)
	return model
