from PIL import Image
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from elpv_dataset.utils import load_dataset
images, proba, types = load_dataset()

def resize_images(size=(128, 128)):
    resized_images = []
    for i in range(images.shape[0]):
        img = Image.fromarray(images[i])
        img_resized = img.resize(size)
        resized_images.append(np.array(img_resized))
    return np.array(resized_images) 

def get_data(normalize='tanh', size=(128, 128)):
    images_128_128 = resize_images(size)
    
    if normalize == 'tanh':
        # Normalize to [-1, 1]
        images_128_128 = (images_128_128 / 127.5) - 1.0
    elif normalize == 'sig':
        # Normalize to [0, 1]
        images_128_128 = images_128_128 / 255.0
        
    y = np.where(proba>0.5,1,0)
    return images_128_128, y, types, proba

# Data Augmentation functions
def horizontal_flip(image):
    """Performs a horizontal flip on an image."""
    return np.fliplr(image)

def vertical_flip(image):
    """Performs a vertical flip on an image."""
    return np.flipud(image)

def rotate_180(image):
    """Performs a 180-degree rotation on an image."""
    return np.rot90(image, k=2)


def split_data(X, Y, Z, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits data into Train, Val, Test.
    """

    temp_size = val_size + test_size 
    

    X_train, X_temp, Y_train, Y_temp, Z_train, Z_temp = train_test_split(
        X, Y, Z, test_size=temp_size, random_state=random_state
    )

    test_relative_size = test_size / temp_size

    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(
        X_temp, Y_temp, Z_temp, test_size=test_relative_size, random_state=random_state
    )

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val, X_test, Y_test, Z_test

def augment_data(X, Y, Z):
    """
    find the smallest sub class (Y,Z) and apply the three augmentations to all its images, 
    the smallest class reach N images,
    now apply the right number of augmentations to other classes until they also reach N images too.


    Returns:

        Augmented X_train_aug, Y_train_aug, Z_train_aug arrays.
        with only the new images added.

    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    Z = np.where(Z=='mono', 0, 1)  # convert string labels to integers

    # build class tuples and counts

    class_tuples = np.array(list(zip(Y, Z)))
    unique_classes, class_counts = np.unique(class_tuples, axis=0, return_counts=True)

    # index map: class_tuple -> list of indices
    idx_map = {}
    for i, t in enumerate(class_tuples):
        key = (int(t[0]), int(t[1]))
        idx_map.setdefault(key, []).append(i)

    # find smallest class (tuple) and compute target N (smallest class after adding 3 augmentations per image)
    min_idx = np.argmin(class_counts)
    smallest_class = tuple(map(int, unique_classes[min_idx]))
    min_class_count = class_counts[min_idx]
    target_N = min_class_count * 4  # original + 3 augmentations

    augment_fns = [horizontal_flip, vertical_flip, rotate_180]

    X_augmented = []
    Y_augmented = []
    Z_augmented = []

    # For the smallest class: apply all three augmentations to every image

    smallest_indices = idx_map[smallest_class]

    for idx in smallest_indices:
        for fn in augment_fns:
            X_augmented.append(fn(X[idx]))
            Y_augmented.append(Y[idx])
            Z_augmented.append(Z[idx])



    # For every other class: apply augmentations until each reaches target_N

    for cls_arr, count in zip(unique_classes, class_counts):
        cl = tuple(map(int, cls_arr))
        if cl == smallest_class:
            continue

        indices = idx_map[cl]
        needed = target_N - count

        if needed <= 0:
            continue

        # cycle through images and augmentation functions to create the required number

        for i in range(needed):
            src_idx = indices[i % len(indices)]
            fn = augment_fns[i % len(augment_fns)]
            X_augmented.append(fn(X[src_idx]))
            Y_augmented.append(Y[src_idx])
            Z_augmented.append(Z[src_idx])

    return np.stack(X_augmented), np.array(Y_augmented), np.where(np.array(Z_augmented)==0, 'mono', 'poly')

def get_defect(X,Y):
    """
    Returns the images where Y==1
    """
    return X[Y==1]