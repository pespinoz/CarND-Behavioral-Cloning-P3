# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image12]: ./images/distribution_1_resized.png "raw"
[image13]: ./images/distribution_2_resized.png "balanced"
[image1]: ./images/nvidia_architecture.png "Model Visualization" 
[image2]: ./images/center_driving.png "Center Driving"
[image3]: ./images/recovery1.png "Recovery Image"
[image4]: ./images/recovery2.jpg "Recovery Image2"
[image5]: ./images/recovery3.jpg "Recovery Image3"
[image6]: ./images/flipped1.jpg "Normal Image"
[image7]: ./images/flipped2.jpg "Flipped Image"
[image8]: ./images/left_view.jpg "Left Image"
[image9]: ./images/center_view.jpg "Center Image"
[image10]: ./images/right_view.jpg "Right Image"
[image11]: ./images/loss.jpg "Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py:` This file contains the script to create and train the NVIDIA convolutional neural network model, which we replicate almost exactly in this project.
* `drive.py:` This file contains the script for driving the car in autonomous mode. All the preprocessing steps taken outside of the Keras implementation (creating a balanced dataset, cropping, and resizing the images) are replicated here.
* `model.h5:` This file contains a trained convolution neural network inspired in the [NVIDIA implementation](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).  
* `writeup_report.md:` You are reading it!
* `preprocessing.py:` In this file  we take most of the preprocessing steps needed for our dataset. The script creates a balanced dataset, and then crops, and resize the images. Other pre-processing, such as normalization, occur within the Keras model.
* `video.mp4:` A video of the car driving in autonomous mode around Track 1 (approximately 2 laps).
* `helper_functions.py:` This file contains several functions that are used both in model.py and preprocessing.py.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

This loads the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.
####3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network inspired in the NVIDIA model described [in this paper](https://arxiv.org/pdf/1604.07316v1.pdf). The choice is made because the problem they deal with is very similar to ours, where the system learns to drive on areas with or without lane markings, with different visual conditions, **_based only on steering training!_**. The architecture consists of 9 layers: one normalization, five 2-D convolutional, and three fully connected (we are not counting dropout layers here). 

The model is implemented in **_lines 67-81 of `model.py`_**. The input are the images we collected from several laps on the simulator and are transformed into the YUV colorspace  (**_line 49_**) before the first normalization layer (**_line 68_**). Feature extraction is performed in the five convolutional layers; first three have 5x5 kernels and 2x2 stride, while the last two are 3x3 kernels and non-strided (**_lines 69-73_**). Each of these layers include ReLU activation functions to introduce non-linearity in the model. Finally there are three fully-connected layers (**_lines 76, 78, 80_**) which act as a controller for the steering angle (output, **_line 81_**).


#### 2. Attempts to reduce overfitting in the model

The model includes three additional [dropout layers](https://arxiv.org/pdf/1207.0580.pdf), which are not present in the original NVIDIA architecture. This regularization technique helps to reduce the overfitting problem effectively (**_`model.py`, lines 75, 77, 79_**). The keeping probability is a hyper-parameter we tuned, starting at the value suggested in Udacity lectures (0.5). We found that keep_prob=0.35 allows for a better performance of the network. 

We must be careful as to how many epochs we want included in our network during training. As we increase this hyperparameter we can obtain better and better training losses, but usually the validation loss we'll reach a minimum and then start increasing again. This is a sign of overfitting. We set the number of epochs to 10 to prevent this problem.

Finally, training and validation were performed on a dataset that contained several modes of driving to ensure generalization of the model. Most of the preprocessing steps performed on the dataset were applied in **_`preprocessing`.py_**, while others such as YUV colorspace transformation and normalization were done in **_`model.py`_** (**_lines 49 and 68, respectively_**). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for at least two laps (_**video.mp4**_).

#### 3. Model parameter tuning

The following parameters were fine-tuned by hand, trying to assess the effect of changes while testing results in AWS EC2 instances. This might be okay for qualitative purposes, however the effect of varying several parameters might be correlated. This is not an optimized process, and furthermore is model architecture-dependent.

Having said that, the parameters of our model that required tuning were the following: 
* Number of epochs: As explained above, after running the model a few times for a large number of epochs, we decided to set this parameter to 10 in order to prevent overfitting (**_line 16 of `model.py`_**). 
* Steering correction: This is the adjustment of steering measurements for the side camera images. After trying values as little as 0.15 and as large as 0.25, we set this parameter to 0.2 (**_line 17 of `model.py`_**) as it gave reasonable results for teaching the car to veer to the center when it started drifting.
* Keeping probability: As explained above this parameter relates to the additional dropout layers introduced in our architecture to combat overfitting. Trying values as low as 0.25 and as high as 0.5, we set this parameter to 0.35 (**_line 19 of `model.py`_**):

Other parameters that didn't require significant fine-tuning were:
* Learning rate: As the model used an adam optimizer to minimize the mean squared error, the learning rate was not tuned manually (**_`model.py` line 85_**).
* Cropping portions of the images: These were used to remove non-essential features such as the sky and the hood of the car. We cropped 70 pixels from the top and 20 pixels from the bottom (**_line 14 of `model.py`_**).
* Test size: We set an 80/20 split between training and validation datasets (**_line 18 of `model.py`_**). 
* Generator's batch size (=32) which specifies the number of samples that are fed into the model at a given time (**_line 32 of `model.py`_**).
 
#### 4. Appropriate training data

Training data was collected from the simulator (track 1) to help our model keep the vehicle driving on the road. During data collection I used the keyboard as an input for the steering as I found the alternative (mouse) very challenging to handle. 

A combination of several driving strategies and the addition of Udacity data were combined to both i) maximize the number of samples for network training, and ii) help generalize the model. Driving strategies include: center lane driving, recovering from the left and right sides of the road, driving clockwise and counterclockwise, and both taking smooth corners and repeating tricky parts of the road (e.g. right after the bridge section). 

For details about how I created the training data, please see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to recognize the success convolutional neural networks have had dealing with feature extraction in images. Computer vision problems such as the one presented in this project are indeed well suited for these kind of networks. Not to mention the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) was built for essentially the same problem as ours, where the system learns to drive on areas with or without lane markings, at different visual conditions, based only on steering training.

Having said this, my first step was to actually try the LeNet network due to its simplicity and just to make sure I could go through the whole process from beginning to end. With a very limited dataset (one lap) and this not-well-suited architecture, I split my image and steering angle data into a training and validation set to gauge how well the model was working. Unsurprisingly, I found that the obtained low mean squared errors reflected underfitting during training.

Iterative changes yielded successive progress. First I added a normalization layer in the Keras model, then data augmentation by flipping images (and inverting steering angles). Then, I implemented changes in the code so I could include the three camera views per measurement. Slowly my dataset began to grow and it was necessary for me to implement a generator function to load images into the model in manageable batches for my machine.

Next changes included the implementation of a convolution neural network matching the NVIDIA model. I found I was getting low mean squared errors on the training set but higher losses on the validation set. This implied that the model was overfitting. To combat it, I modified the NVIDIA model to include regularization in the form of dropout layers that followed the original fully-connected layers. I was obtaining much better numerical results regarding losses (both in training and validation), but the simulator testing was still poor.

Then I included other pre-processing steps such as the conversion to YUV colorspace, cropping and resizing of the images to the size expected by the NVIDIA architecture. Testing continued to improve.

Parallel to this whole development my data collecting has increased considerably. This fact together with using a much more complex architecture made me migrate my project to an AWS instance where I could use a GPU to train my network. The last singular improvement I made in my preprocessing came from examining the distribution of steering angles in my complete dataset. This is its histogram:

![alt text][image12] 

Clear peaks are seen around 0, -0.2, and 0.2. This underlines the fact that straight driving is extremely overrepresented in our dataset. To improve this situation and obtain a more balanced dataset I randomly sampled the images where the car was driving straight, keeping only a 20% of them. This resulted in a significantly smaller dataset (bad), but a much more balanced one (good!). 

![alt text][image13]

tuning
 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle drove too close to the side of the road, and further improvement included the fine-tuning of my model parameters (discussed in the previous Section).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road (see _**`video.mp4`**_, a [recording](./video.mp4) of the car driving in autonomous mode around Track 1 approximately for 2 laps).

####2. Final Model Architecture

The final model architecture (**_lines 67-81 of `model.py`_**) is based on the NVIDIA convolutional neural network described [here](https://arxiv.org/pdf/1604.07316v1.pdf). It receives as input 66x200 pixels images in the YUV colorspace. The architecture consists of one normalization layer (values set between -0.5 to 0.5), five 2-D convolutional layers (first three have 5x5 kernels and 2x2 stride, while the last two are 3x3 kernels and non-strided), and finally three fully connected layers for the regression stage. 

Each of these convolutional layers include ReLU activation functions to introduce non-linearity in the model. In addition to this we added dropout layers after the fully connected layers as a regularization technique to prevent overfitting. There are 252219 total trainable parameters, and more than 27 million connections in the model: A more detailed description is summarized in the following Table.


| Layer         		|     Description	        					|  Output Shape | Number of Parameters     |
|:---------------------:|:---------------------------------------------:|:-------------:|:-------------------------|
| Layer 1 |  Normalization from (0,255) to (-0.5,0.5) range        | 66x200x3      | 0                        |
| Layer 2 |  2D Convolution (5x5 filter with depth=24, and 2x2 stride)  | 31x98x24      | 1824                     |
|         |  ReLU activation                                            |               |                          |
| Layer 3 |  2D Convolution (5x5 filter with depth=36, and 2x2 stride   | 14x47x36      | 21636                    |
|         |  ReLU activation                                            |               |                          |
| Layer 4 |  2D Convolution (5x5 filter with depth=48, and 2x2 stride   | 5x22x48       | 43248                    |
|         |  ReLU activation                                            |               |                          |
| Layer 5 |  2D Convolution (3x3 filter with depth=64, and no stride    | 3x20x64       | 27712                    |
|         |  ReLU activation                                            |               |                          |
| Layer 6 |  2D Convolution (3x3 filter with depth=64, and no stride    | 1x18x64       | 36928                    |
|         |  ReLU activation                                            |               |                          |
|         |  Flatten                                                    | 1152          | 0                        |
| Add'tl Layer |   Dropout (keep_prob = 0.35)                           | 1152          | 0                        |
| Layer 7 |  Fully-connected                                            | 100           | 115300                   |
| Add'tl Layer |   Dropout (keep_prob = 0.35)                           | 100           | 0                        |
| Layer 8 |  Fully-connected                                            | 50            | 5050                     |
| Add'tl Layer |   Dropout (keep_prob = 0.35)                           | 50            | 0                        |
| Layer 9 |  Fully-connected                                            | 10            | 510                      |
| Output  |  Fully-connected                                            | 1             | 11                       |


Finally, here is a visualization of the architecture (taken from Figure 4 of the [Bojarski et al. 2016 paper](https://arxiv.org/pdf/1604.07316v1.pdf)).

![alt text][image1]

####3. Creation of the Training Set & Training Process

The goal of data collection in the simulator is to 
to capture good driving behavior so the model learns it, and then the autonomous mode is able to predict steering angles satisfactorily.

* i) First, I recorded three laps on track one using center lane driving in the default mode (counter-clockwise). This process captured 3849 images. Here is an example image of center lane driving:

![alt text][image2]

* ii) Second, to help generalize the model, I drove two more laps but this time they were clockwise. In the default mode (counter-clockwise) all corners but one ae left corners, so this step helps to build a more balanced dataset. Here, we captured 2544 more images.

* iii) Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the middle once it deviated from center lane driving. I captured 1148 of these images. The next series of figures show what a recovery looks like starting from the right curb:

![alt text][image3]
![alt text][image4]
![alt text][image5]

* iv) In the fourth step, I captured a few more images for "tricky" corners in which my model had previously presented difficulties during autonomous mode. This was the less significant addition to the dataset, with only 660 images.

* v) Finally, I took advantage of the training set provided by Udacity to augment the final dataset. I just merged it to my own data collection. The contribution of the Udacity dataset was by far the largest, being comprised by more than 8000 images.
-----------------
Now for the augmentation stage:
* To augment the dataset, I also flipped images (and consequently inverted steering angles). This also helps generalize the model as most of the time it's learning to steer to the left. Then, in autonomous mode car continues to steer to the left even if going in a straight line is better. To mitigate this behaviour a balanced dataset obtained by flipping images and inverting angles is needed. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

This step multiplies our samples x2!

* Also we took advantage of the fact we have three images per measurement, corresponding to left, center, and right cameras. They all capture the same scene but from slightly different positions. Adding these images carries a two-fold benefit: i) more data to train the network, and ii) data we use is more comprehensive as it teaches the network what is to be off-center of the road and how to steer back to the middle. For example, here are the different perspectives of the same measurement:

![alt text][image8]
![alt text][image9]
![alt text][image10]

This step multiplies our samples x3!

-----------------

After collection, augmentation, and the random sampling to create a more balanced dataset (described in the "Solution Design Approach" Section) I had 43668 data points. I then preprocessed this data by:
* Cropping the images to remove non-essential features such as the sky and the hood of the car. We cropped 70 pixels from the top and 20 pixels from the bottom (**_line 102 of `preprocessing.py`_**).
* Resizing the images to the shape expected by the NVIDIA architecture, which is 200x66 (**_line 107 of `preprocessing.py`_**).
* Transformation of the images to the YUV colorspace to optimize feature extraction in the convolutional layers. This was easier to do when loading the images in our generator function (**_line 49 of `model.py`_**).
* Normalization and centering pixel values in the (-0.5, 0.5) range. This was easier to do in the first layer of the Keras model (**_line 68 of `model.py`_**).

-----------------

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used the remaining 80% of the data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. As the dataset was considerably large (at least for my machine capabilities) I had to use a generator function to load and train batches of	data (size=32). In this way storing all data in memory at
once was unnecessary (see **_line 32 of `model.py`_**).

The ideal number of epochs was 10, as evidenced by the following Figure showing training and validation losses. This particular saved model (_**`model.h5`**_) drives the track successfully, with smooth corners and mostly center-lane driving. This is also shown in _**`video.mp4`**_, a [video](./video.mp4) of the car driving in autonomous mode around Track 1 approximately for 2 laps.

![alt text][image11]
