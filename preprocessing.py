import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import skimage.feature as skif


def lbp_histogram(image, P=8, R=1, method='nri_uniform'):
    lbp = skif.local_binary_pattern(image, P, R, method)
    max_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return hist


def extract_feature(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_h = lbp_histogram(image[:, :, 0])  # y channel
    cb_h = lbp_histogram(image[:, :, 1])  # cb channel
    cr_h = lbp_histogram(image[:, :, 2])  # cr channel
    feature = np.concatenate((y_h, cb_h, cr_h))
    np.save('new_img_feature.npy', np.array(feature))


def load_feature(file_name):
    features = np.load(file_name)
    return features
