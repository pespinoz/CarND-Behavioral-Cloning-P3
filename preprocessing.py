"""
Let's say I've obtained X images from the simulator. But just want to take into account the first 10000 images. And from
these, build a more balanced dataset which will be stored at a directory called "data." Then, this script is called from
the command line as:
python preprocessing.py data 1,10000
"""

import sys
import time
from shutil import copyfile
from helper_functions import *

NVIDIA_SIZE = (66, 200, 3)  # shape of the images expected by NVIDIA model
CROP_Y = (70, 140)  # crop $1 pixels from top; crop everything down of $2 pixels
CROP_X = (0, 320)   # crop $1 pixels from left; crop everything right of $2 pixels
CORRECTION = 0.2    # adjusted steering measurements for the side camera images
SAMPLING = 0.10     # sampled fraction of the central bin in steer-angle histogram. Helps create a more balanced dataset
CENTRAL_BIN = 0.015  # defines the central bin mentioned above.
IN_DIR = '../p3-data/data_all/'  # path to where all the simulator images are stored
IN_DIR_IM = '../p3-data/data_all/IMG/'  # (see line 1 for an exaplanation of how this script works)

lim_inf, lim_sup = [], []
for arg in sys.argv:
    if arg != sys.argv[0] and arg != sys.argv[1]:
        lim_inf.append(int(arg.split(',')[0]))
        lim_sup.append(int(arg.split(',')[1]))

# These created the output tree directory structure where the (balanced) training set will be stored (see line 1 and 19)
OUT_DIR = '../p3-data/' + sys.argv[1] + '/'
OUT_DIR_IM = '../p3-data/' + sys.argv[1] + '/IMG/'
OUT_DIR_IM_PREPROC = '../p3-data/' + sys.argv[1] + '/IMG_PREPROC/'
create_or_rewrite(OUT_DIR)
create_or_rewrite(OUT_DIR_IM)
create_or_rewrite(OUT_DIR_IM_PREPROC)

########################################################################################################################

# First I read the filenames for all the images obtained from the simulator.
_, centers, lefts, rights, _ = read_csvfile(IN_DIR + 'driving_log.csv')

# But since I'm calling these script as python preprocessing.py data 1,n, I'm interested only in the first n-ones.
# Therefore I filter the previous csv file to reflect this requirement.
for j in range(len(lim_sup)):
    cmnd = 'sed -n ' + str(lim_inf[j]) + ',' + str(lim_sup[j]) + 'p ' + IN_DIR + 'driving_log.csv' + \
           ' >> ' + OUT_DIR + 'driving_log.csv'
    os.system(cmnd)

# And correspondingly I copy the chosen n-images to the first output directory.
for u, v in zip(lim_inf, lim_sup):
    for t in range(u, v+1):
        copyfile(IN_DIR_IM + centers[t-1].split('/')[-1], OUT_DIR_IM + centers[t-1].split('/')[-1])
        copyfile(IN_DIR_IM + lefts[t-1].split('/')[-1], OUT_DIR_IM + lefts[t-1].split('/')[-1])
        copyfile(IN_DIR_IM + rights[t-1].split('/')[-1], OUT_DIR_IM + rights[t-1].split('/')[-1])
        print(t)
del centers, lefts, rights

# Next two lines give me the histogram of the steering angle distribution for these n-images. It's highly unbalanced, as
# expected, since the car mostly follows straight lines and therefore there's a peak in the central bin.
# get_angle_hist() is defined in helper_functions.py
lines, centers, lefts, rights, angle_steering = read_csvfile(OUT_DIR + 'driving_log.csv')
angles = get_anglehist(angle_steering, CORRECTION, 'Steering Angle Distribution', '1')

########################################################################################################################

# To help balance the dataset I sample the central bin, and only keep a SAMPLING fraction of it.
inds = np.squeeze(np.array(np.where(np.abs(np.array(angle_steering)) < CENTRAL_BIN)))
inds_rest = np.squeeze(np.array(np.where(np.abs(np.array(angle_steering)) >= CENTRAL_BIN)))
inds_selected = np.concatenate([np.random.choice(inds, int(SAMPLING*inds.shape[0])), inds_rest])

# I filter images, filenames, and steering angles correspondingly:
newlines = [lines[ind] for ind in inds_selected]
newcenters = [centers[ind] for ind in inds_selected]
newlefts = [lefts[ind] for ind in inds_selected]
newrights = [rights[ind] for ind in inds_selected]
newangle_steering = [angle_steering[ind] for ind in inds_selected]

# And again obtain the histogram for the steering angle distribution. Although not ideal, the situation regarding how
# balanced is the dataset is much improved.
newangles = get_anglehist(newangle_steering, CORRECTION, 'Filtered Steering Angle Distribution', '2')

########################################################################################################################

# Now that we have filtered the central bin of the original distribution to obtain a more balanced dataset, we take
# pre-processing steps on these images.
with open(OUT_DIR + 'driving_log_preproc.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for line in newlines:
        writer.writerow(line)

# First we load the images and their filenames:
t0 = time.time()
fids = newcenters + newlefts + newrights
filenames = [name.split('/')[-1] for name in fids]
images = []
for filename in filenames:
    im = cv2.imread(OUT_DIR_IM + filename)
    images.append(im)
print("Loading Time: %.3f seconds" % (time.time() - t0))

# Then we crop them:
t1 = time.time()
cropped = [image[CROP_Y[0]:CROP_Y[1], CROP_X[0]:CROP_X[1]] for image in images]
print("Cropping Time: %.3f seconds" % (time.time() - t1))

# And finally we resize them to the shape the NVIDIA model expects.
t2 = time.time()
resized = [cv2.resize(image, (NVIDIA_SIZE[1], NVIDIA_SIZE[0]), interpolation=cv2.INTER_CUBIC) for image in cropped]
print("Resizing Time: %.3f seconds" % (time.time() - t2))

# We write these images into OUT_DIR_IM_PREPROC/. This is the location that will feed model.py
t3 = time.time()
[cv2.imwrite(OUT_DIR_IM_PREPROC + filename, image) for image, filename in zip(resized, filenames)]
print("Writing Time: %.3f seconds" % (time.time() - t3))
