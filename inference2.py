from model.srgan import generator
from utils import load_image, plot_sample
from model import resolve_single
import tensorflow as tf
import numpy as np
import os
from PIL import Image

model = generator()
model.load_weights('gan_generator.h5')

TEST_DIR = "testing_lr_images/testing_lr_images/"
ANS_DIR = "ans/"
testing_imgs = os.listdir(TEST_DIR)
if not os.path.exists(ANS_DIR):
    os.mkdir(ANS_DIR)

for img_path in testing_imgs:
    full_path = os.path.join(TEST_DIR, img_path)
    input_img = load_image(full_path)
    result = resolve_single(model, input_img)
    result = tf.image.resize(result, (input_img.shape[0]*3, input_img.shape[1]*3))

    result = Image.fromarray(result.numpy().round().astype(np.uint8)).convert('RGB')
    result.save(ANS_DIR+f"{img_path[:-4]}_pred.png")
