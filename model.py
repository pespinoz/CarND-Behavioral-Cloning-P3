import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from helper_functions import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')   # if executing in AWS include this line; in my machine I can comment it
plt.interactive(False)

CSV_PATH = '../p3-data/data/driving_log_preproc.csv'    # csv file listing the pre-processed images and steering angles
IM_PATH = '../p3-data/data/IMG_PREPROC/'                # path to folder that contains pre-processed images
PLOTS_PATH = './images/'                                # patht to output plots
SAVED_MODEL = 'model.h5'                                # path to the trained convolution neural network
EPOCHS = 10       # number of epochs in which my model is trained.
CORRECTION = 0.2  # adjusted steering measurements for the side camera images
TEST_SIZE = 0.20  # training / validation split. Currently 80/20 respectively
KEEP_PROB = 0.35  # keep probability for the dropout layers added to the NVIDIA model.
TIMES = 6         # augmentation: using left, center, right cameras (x3) and flipping images (x2)
BATCH_SIZE = 32   # batch size that feeds into the model at a given time.
NVIDIA_SIZE = (66, 200, 3)                              # shape of the images expected by NVIDIA model
SCREENSIZE = (16, 8.5)                                  # size of my screen (for plot-purposes)

########################################################################################################################

# Read the global dataset csv file. From lines I'll extract filenames for images and steering angles.
lines, _, _, _, _ = read_csvfile(CSV_PATH)

# I divide lines (filenames for each central, left, right images) between training and validation sets.
train_samples, validation_samples = train_test_split(lines, test_size=TEST_SIZE)


def generator(samples, batch_size):
    """
    # Define a generator function that pull pieces of the data and process it on the fly. This way we avoid having to
    store all preprocessed data in memory at once (this is not possible in my machine).
    :param samples: set we will process, either training or validation
    :param batch_size: size of the set that it's fed into the model at a given time.
    :return: shuffled batch of either training or validation data
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            # We get images and their labels (steer angles). get_images_get_labels() is defined in helper_functions.py
            # This step includes the transformation to YUV colorspace (it was easier to implement it here rather than in
            # preprocessing.py)
            images, labels = get_images_get_labels(batch_samples, IM_PATH, CORRECTION)
            # Next we augment the dataset, for the purpose of expanding and generalizing the training.
            # augmentation() is defined in helper_functions.py
            X_train, y_train = augmentation(images, labels)

            yield shuffle(X_train, y_train)

# We will compile and train the Keras model using the generator functions for both training and validation sets.
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

########################################################################################################################

# Here I develop the model architecture within a Keras framework. I imitate the NVIDIA model (more in the writeup), but
# with a few differences. Fundamentally, I'm using Dropout layers after each fully connected layer for the purpose of
# minimizing overfitting. Also the normalization process for the images is implemented here. Again easier to do it here
# rather than in preprocessing.py)
t0 = time.time()
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=NVIDIA_SIZE))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu", input_shape=()))  # Five 2-D convolutional layers
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))                  # with ReLUs activation.
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(KEEP_PROB))  # Additional dropout layers to combat over-fitting
model.add(Dense(100))  # fully connected layer
model.add(Dropout(KEEP_PROB))  # Additional dropout layers to combat over-fitting
model.add(Dense(50))  # fully connected layer
model.add(Dropout(KEEP_PROB))  # Additional dropout layers to combat over-fitting
model.add(Dense(10))  # fully connected layer
model.add(Dense(1))

model.summary()
# The model is trained with an Adam optimizer:
model.compile(loss='mse', optimizer='adam')
# And fed with batches of data that are processed on the fly through the generator function defined above.
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*TIMES,
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples)*TIMES,
                                     nb_epoch=EPOCHS, verbose=1)
model.save(SAVED_MODEL)  # Saves the model parameters to be used in the testing step (autonomous mode in the simulator)

########################################################################################################################

# Here I plot and save the loss as a function of epochs for both training and validation.
# To prevent overfitting the model is trained until validation achieves it's lower loss value.
fig = plt.figure(figsize=(SCREENSIZE[0], SCREENSIZE[1]/2))
plt.plot(range(1, EPOCHS+1), history_object.history['loss'])
plt.plot(range(1, EPOCHS+1), history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
fig.savefig(PLOTS_PATH + 'loss.jpg', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight',
            pad_inches=0.1, frameon=None)
