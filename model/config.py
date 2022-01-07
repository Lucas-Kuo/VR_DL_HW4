# import the necessary packages
import os

# specify original path to the dataset
ORIG_INPUT_DATASET = "training_hr_images/training_hr_images/"

# specify root path to the dataset
ROOT_PATH = "dataset/"

# specify paths to the different splits of the dataset
TRAIN_SET = os.path.join(ROOT_PATH, "train")
VAL_SET = os.path.join(ROOT_PATH, "val")
TEST_SET = "testing_lr_images/testing_lr_images/"

# the ratio of validation images to the number of images
VAL_RATIO = 0.1

# specify the initial size of the images and downsampling factor
ORIG_SIZE = (300, 300)
DOWN_FACTOR = 3

# specify number of RDB blocks, batch size, number of epochs, and
# initial learning rate to train our model
RDB_LAYERS = 3
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3

#define paths to serialize trained model, training history plot, and
# path to our inference visualizations
SUPER_RES_MODEL = os.path.join("output", "super_res_model")
TRAINING_PLOT = os.path.join("output", "training.png")
VISUALIZATION_PATH = os.path.join("output", "visualizations")
ANSWER_PATH = "answer/"
OUT_DIR = "output/"
