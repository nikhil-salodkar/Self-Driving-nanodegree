# Traffic Sign Recognition

## Writeup

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/train_distribution.png "Visualization1"
[image_valid]: ./examples/validation_distribution.png "Visualization2"
[image_test]: ./examples/test_distribution.png "Visualization3"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And the code for this project present in jupyter notebook and the necessary weight files can be found in the Master branch of this repository

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training, validation and test images with different labels are distributed.

training images distribution is :

![train_distribution][image1]

validation images distribution is :

![validation_distribution][image_valid]

test images distribution is :

![test_distribution][image_test]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initiall I tried training the model without performing any normalization steps. But, without normalization of the images the validation accuracy was not going beyond 90% accuracy even after training for many epochs like more than 50. 
Then I tried normalizing the images using Scikit learn's MinMaxScaler function. Then I verified if normalizing the images is not causing any issues to image by visualizing the images using OpenCV. After normalizing as you can see the model started training faster and also reached accuracy above 94% on validation data.

Although I know that I could perform data augmentation techniques like horizontal flipping which would definitely generate more data and could improve the overall accuracy and even help in correct prediction of images taken in the wild from the Internet, I chose not to do that as I got the required accuracy without performing data augmentation. Anyway, performing data augmentation has been made easier using new Tensorflow versions rather than this version.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5X5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14X14x6 				|
| Dropout(0.1)  |     Randomly drop 10% of the next nodes  |
| Convolution 5X5    | 1X1 stride, valid padding, outputs 10X10X16  |
| RELU |           |
| Max pooling         | 2X2 stride, outputs 5X5X16  |
| Dropout(0.1)  |    Randomly drop 10% of the next nodes|
| Flatten  | Flatten the output to connect to fully connected layer, output 400| 
| Fully connected		| output 120        									|
| RELU |        |
| Dropout(0.5)   |Randomly drop 50% of the next nodes  |
| Fully connected  |output 84  |
| RELU |       |
| Dropout(0.5)    |Randomly drop 50% of the next nodes  |
| Softmax				| output 43(number of classes)       									|

While finalizing the architecture, good additions to the architectures are the inclusions of dropout layers, Adding the dropout layers helped in reducing the overfitting which was happening without adding them. Also, two different dropout values were used, much less dropout(0.1) for convolutional layers and higher dropout (0.5) for fully connected layers. The intuition behind this is we need the model to learn as much pattern as possible using Convolutional layers and their filter but more generalize the fully connected layers with higher dropout because the spacial information is lost when we convert the convolutions to dense layers.
