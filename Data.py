# Note from the author: This file is to convert the images in hdf5 format to accelerate the loading process,
# Users should rewrite this file to load your own data including the 'compressData()' and load_data1/2()'.

import numpy as np
from skimage import io
import argparse
import h5py

def load_data(txt_name):
    path = open('/home/dl/zyfang/face_example/'+txt_name, 'r')
    lines = path.readlines()
    X_train = []
    for i in range(len(lines)):
        img = io.imread(lines[i].replace('/home/zyfang/caffe-face', '/home/dl/zyfang').replace(' 0\n', ''))
        if (len(img.shape) == 3):
            X_train.append(img)
    return np.asarray(X_train)

def load_data2(txt_name):
    path = open('/home/dl/zyfang/face_example/'+txt_name, 'r')
    lines = path.readlines()
    X_train = []
    for i in range(len(lines)):
        t = lines[i].replace('/media/sdb/ECCV16-SIAT/','/home/dl/zyfang/')
        t = t[0: t.index('.jpg')+4]
        img = io.imread(t)
        if (len(img.shape) == 3):
            X_train.append(img)
    return np.asarray(X_train)

def compressData():
    X_train = load_data('lr_training.txt')
    y_train = load_data2('hr_training.txt')
    X_test = load_data('lr_val.txt')
    y_test = load_data2('hr_val.txt')
    # Create the HDF5 file
    f = h5py.File('data.h5', 'w')

    # Create the image and palette dataspaces
    dset = f.create_dataset('lr_train', data=X_train)
    pset = f.create_dataset('hr_train', data=y_train)
    lset = f.create_dataset('lr_test', data = X_test)
    sset = f.create_dataset('hr_test', data = y_test)

    # Close the file
    f.close()

def loadData(path):
    f = h5py.File(path, "r")
    data = [f['lr_train'][:], f['hr_train'][:], f['lr_test'][:], f['hr_test'][:]]
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--path", type=str, default='data.h5')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "compress":
        compressData()
        print('File stored as "data.h5" in current directory.')
    elif args.mode == "load":
        loadData(args.path)
        print('File loaded from '+args.path)