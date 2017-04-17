# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based closely on the model described in the NVIDIA paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf). It employs a series of convolutional layers paired with ReLUs. The layers (lines 112 - 134) are:

Layer | Type           | Specification
------|----------------|----------------------------------------
0 | Input | 320 w x 160 h RGB Image
1 | Lambda | Normalization (zero-centering) all pixels, channels
2 | Crop | Crop top 70 and bottom 15, result 320 x 75
3 | Convolution | 5x5 kernel, 2x2 strides, 24 depth
4 | ReLU | 
5 | Convolution | 5x5 kernel, 2x2 strides, 36 depth
6 | ReLU | 
7 | Convolution | 5x5 kernel, 2x2 strides, 48 depth
8 | ReLU | 
9 | Dropout | 0.25 probability
10 | Convolution | 3x3 kernel, 64 depth
11 | ReLU | 
12 | Convolution | 3x3 kernel, 64 depth
13 | ReLU | 
14 | Dropout | 0.25 probability
15 | Flatten | 
16 | FC | 1164 output
17 | ReLU | 
18 | Dropout | 0.5 probability
19 | FC | 100 output
20 | ReLU | 
21 | FC | 50 output
22 | ReLU | 
23 | FC | 10 output
24 | ReLU | 
25 | FC | 1 output

Before I used this model, I also tried a network based on my LeNet implementation and a model based on VGG-16. Both models were able to drive Track 1. However, both models became very large with the input sizes of this problem and were slow to train. They also tended to perform poorly on Track 1 if they were trained with data on Track 2. When I switched to this model, it was much faster to train (due to the strides in the first 3 convolutional layers). However I remained with RGB channels instead of HSL/HSV as described in the NVIDIA paper.

 
#### 2. Attempts to reduce overfitting in the model

Overfitting was a significant problem when I added track 2 data. 

I kept the same dropout structure as VGG-16/19, with dropouts at the ends of both sets of convolutional layers and at the first FC layer. The NVIDIA model does not mention the VGG models, but it looks closely related to them, so I assumed that a similar dropout structure would work.

I also trained the flipped images (lines 45-47). 

I trained the model for 10 epochs but made checkpoints at each epoch (line 138), this allowed me to test different epochs.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 142).

#### 4. Appropriate training data

I generated vertically flipped images from each camera, and used all three camera positions (left, center, right) and tuned the side camera outputs. 

I trained both tracks in the same model. I trained three types of data: centered driving, "recovery driving", and recordings of specific turns. The latter was the majority of data in the final model, because to succeed at Track 2, I had to repeatedly drive around the tightest curves on the track in order to get the model to train for the highest steering angles possible. I also found that recording the runs at extremely low speeds resulted in more training data through the corners.

I also attempted some other training data strategies which did not result in better models. The first was to record data on Track 1 where I drove the car close to one yellow line. My theory was that this might be closer to the Track 2 data, which has tighter lane markings, but this resulted in poorer models. I also recorded data in Track 2 where I drove on the center line, because I hoped to train the model not to stop turning as it was crossing the center line in tight corners. However this resulted in very bad performance on pretty much every track, so also removed this data set.

I am not sure how much of the data is track 1 vs track 2 because I only started separating them near the end of training.

My final data set was 447870 input images, which included all side cameras and flipped images.

### Changes to drive controller
In order to drive Track 2 in autonomous mode, I modified the drive controls from their default. I tuned the constants on the PI throttle controller to lower the Ki term so that it did not overshoot as much after cresting hills. I introduced linear reduction in speed based on the turning angle, which gave the model more time to give steering inputs, necessary for the tightest corners. Finally I worked around a bug where the car would become stuck if it was driving at very low speeds through a terrain change (I believe the car was "bottoming out"). The workaround to this was to randomly apply the brakes if the car was not moving.

### Results

My model is able to drive on Track 1 and Track 2. The performance on Track 1 degraded as more of the training data was focused on Track 1. I believe that more center line data for Track 1 would have corrected the two areas, but I did not want to continually attempt to optimize between the two tracks, and the model took several hours to train.

There are two areas of Track 1 that presented problems: at the transition from the bridge back to the road, and at a specific point where there is some brush ahead of the road. Both caused the model to steer aggressively to the left. I added more data from these areas to try to correct the model.

There were many many areas of Track 2 that presented problems, but the most difficult were the tightest turns. To deal with these, I introduced the throttle changes, created reams of training data of both lanes, and used the "fastest" setting at 640x480 on the sim. The fastest setting reduces noise and does not cast shadows, which made the model easier to train, so it is a little bit of a cheat. However, I did it because I wanted to guarantee that the model got as many frames as possible and was able to steer the car as quickly as possible.

I also tested my model on both tracks going the opposite way. The model was able to drive Track 1, but drove off the road on Track 2 at the double hairpin curve. If I skipped this curve manually, it drove the rest of the track. Since these curves are by far the biggest challenge, I suspect that I would be able to drive the entire track if I gave more training data.