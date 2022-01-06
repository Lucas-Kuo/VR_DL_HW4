#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model import config
from random import shuffle
import os
import shutil

# listing all image file names
imagePaths = os.listdir(config.ORIG_INPUT_DATASET)

# randomly shuffle the paths
shuffle(imagePaths)

# create the training dataset directory if not exist
if not os.path.exists(config.TRAIN_SET):
    print("[INFO] Creating {} directory...".format(config.TRAIN_SET))
    os.makedirs(config.TRAIN_SET)

# create the validation dataset directory if not exist
if not os.path.exists(config.VAL_SET):
    print("[INFO] Creating {} directory...".format(config.VAL_SET))
    os.makedirs(config.VAL_SET)

# split for validation
for imagePath in imagePaths[:int(len(imagePaths)*config.VAL_RATIO)]:
    old_dir = os.path.join(config.ORIG_INPUT_DATASET, imagePath)
    new_dir = os.path.join(config.VAL_SET, imagePath)
    shutil.copy2(old_dir, new_dir)

# split for training
for imagePath in imagePaths[int(len(imagePaths)*config.VAL_RATIO):]:
    old_dir = os.path.join(config.ORIG_INPUT_DATASET, imagePath)
    new_dir = os.path.join(config.TRAIN_SET, imagePath)
    shutil.copy2(old_dir, new_dir)
