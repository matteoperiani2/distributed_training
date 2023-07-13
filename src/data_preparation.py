import os
import random
import numpy as np
import idx2numpy
import cv2 
import matplotlib.pyplot as plt

def get_data(folder_path):
    x_train = idx2numpy.convert_from_file(os.path.join(folder_path, "train-images"))
    y_train = idx2numpy.convert_from_file(os.path.join(folder_path, "train-labels"))
    x_test = idx2numpy.convert_from_file(os.path.join(folder_path, "test-images"))
    y_test = idx2numpy.convert_from_file(os.path.join(folder_path, "test-labels"))
    return (x_train, y_train), (x_test, y_test)

# transform into binary task
def data_preprocessing(y_train, y_test, selected_category):
    y_train = np.where(y_train == selected_category, 1, 0)
    y_test = np.where(y_test == selected_category, 1, 0)
    return y_train, y_test

# reduce number of images of train/test dataset balancing them
def reduce_dataset(x_train, y_train, x_test, y_test, train_size, test_size):
    idx_ones = random.choices(np.where(y_train == 1)[0], k=int(train_size/2))
    idx_zeros = random.choices(np.where(y_train == 0)[0], k=int(train_size/2))
    train_idx = idx_ones + idx_zeros
    random.shuffle(train_idx)
    idx_ones = random.choices(np.where(y_test == 1)[0], k=int(test_size/2))
    idx_zeros = random.choices(np.where(y_test == 0)[0], k=int(test_size/2))
    test_idx = idx_ones + idx_zeros
    random.shuffle(test_idx)
    return x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]

def prepare_dataset(folder_path, selected_cat, train_size, test_size, size):
    (x_train, y_train), (x_test, y_test) = get_data(folder_path)
    
    print("Data loaded!")
    print(f"Train data shape: x={x_train.shape}, y={y_train.shape}")
    print(f"Test data shape: x={x_test.shape}, y={y_test.shape}")

    print("\nStart preprocessing...")
    selected_cat = selected_cat
    print(f"Selected clothes: {np.unique(y_train)[selected_cat]}")
    y_train, y_test = data_preprocessing(y_train, y_test, selected_category=selected_cat)
    print("Binarization of the dataset completed!")
    
    print("Reduncing datasets dimensions:")
    x_train, y_train, x_test, y_test = reduce_dataset(x_train, y_train, x_test, y_test, train_size=train_size, test_size=test_size)
    print(f"Train data shape: x={x_train.shape}, y={y_train.shape}")
    print(f"Test data shape: x={x_test.shape}, y={y_test.shape}")
    print(f"Number of sneaker images on train split: {np.sum(y_train)} ({np.sum(y_train)/y_train.shape[0]*100:.2f} %)")
    print(f"Number of sneaker images on test split: {np.sum(y_test)} ({np.sum(y_test)/y_test.shape[0]*100:.2f} %)")

    ## downsampling image resolution
    print("\nDownsampling the datasets to better speed performance:")
    preprocess = lambda x: cv2.resize(x, size).flatten() / 255.
    x_train = np.array([preprocess(x) for x in x_train])
    x_test  = np.array([preprocess(x) for x in x_test])
    print(f"Train data shape: x={x_train.shape}, y={y_train.shape}")
    print(f"Test data shape: x={x_test.shape}, y={y_test.shape}")

    return x_train, y_train, x_test, y_test

def split_images_per_agents(images, labels, n_agents, imgs_per_agent):
    rng = np.random.default_rng()
    idx_ones = np.random.choice(np.where(labels == 1)[0], size=(n_agents, int(imgs_per_agent/2)), replace=False)
    idx_zeros = np.random.choice(np.where(labels == 0)[0], size=(n_agents, int(imgs_per_agent/2)), replace=False)
    agents_idxs = np.concatenate((idx_ones, idx_zeros), axis=1)
    rng.shuffle(agents_idxs, axis=1)
    # agents_idxs = np.random.choice(images.shape[0], size=(n_agents, imgs_per_agent), replace=False)
    imgs = []
    lbls = []
    for idxs in agents_idxs:
        imgs.append(np.array(images[idxs]))
        lbls.append(np.array(labels[idxs]))
    
    return np.array(imgs), np.array(lbls)
