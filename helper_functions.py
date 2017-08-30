import csv
import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.interactive(True)

# The following are helper functions I wrote to make the model.py and preprocessing.py scripts more simple:


def load_images(folder):
    """
    It loads all images in a given folder (using cv2). It reports how many were actually loaded, and return images and
    its filenames.
    """
    loaded_images, loaded_filenames = [], []
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            im = cv2.imread(os.path.join(folder, filename))
            if im is not None:
                loaded_images.append(im)
                loaded_filenames.append(filename)
    print("Loaded {} images".format(len(loaded_images)))
    return loaded_images, loaded_filenames


def create_or_rewrite(folder_name):
    """
    Creates a given directory, and if it already exists, it re-writes it.
    """
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)


def read_csvfile(path):
    """
    Custom function that reads the csv files we're interested in this projects. It returns the lines as a whole, as well
    as the filenames for the center, left, and right cameras. It also returns the steering angles (as floats).
    """
    lines, centers, lefts, rights, angles = [], [], [], [], []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            centers.append(line[0])
            lefts.append(line[1])
            rights.append(line[2])
            angles.append(float(line[3]))
    return lines, centers, lefts, rights, angles


def get_images_get_labels(batches, images_path, angle_correction):
    """
    Gets images and labels (steering angles) for each batch within a generator function defined in model.py
    """
    images = []
    labels = []
    for batch_sample in batches:
        for k in range(3):  # center, left and right cameras
            current_path = images_path + batch_sample[k].split('/')[-1]  # fixes path for AWS EC2
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            images.append(image)

        steering_center = float(batch_sample[3])
        steering_left = steering_center + angle_correction
        steering_right = steering_center - angle_correction
        labels.extend([steering_center, steering_left, steering_right])
    return images, labels


def augmentation(in_images, in_labels):
    """
    Performs the data augmentation step. This is done by flipping images; effectively doubling the number of examples
    available for the training of the model.
    """
    augmented_images, augmented_labels = [], []
    for image, label in zip(in_images, in_labels):
        augmented_images.append(image)
        augmented_labels.append(label)
        augmented_images.append(cv2.flip(image, 1))
        augmented_labels.append(label * -1)

    xtrain = np.array(augmented_images)
    ytrain = np.array(augmented_labels)
    return xtrain, ytrain


def get_anglehist(asteering, correction, string, string2):
    """
    Plots (and save) the histogram of the steering angles distribution. This is done after considering that the three
    cameras (left, center, right), and the data augmentation step.
    """
    a3cams = [angle for angle in asteering] + \
         [angle + correction for angle in asteering] + \
         [angle - correction for angle in asteering]
    a3cams_flipped = [-1*angle for angle in a3cams]
    a = a3cams + a3cams_flipped

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    colors = ['lime']
    ax0.hist(a, 50, normed=1, histtype='bar', color=colors, label=['Dataset'])
    ax0.legend(prop={'size': 10}, loc='upper right', frameon=False)
    ax0.set_xlabel(string)
    ax0.set_ylabel('Ocurrences (normalized)')
    fig.savefig('images/distribution_' + string2 + '.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
    # plt.close(fig2)

    return a
