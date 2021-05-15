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
[internet_image1]: ./examples/Internet_images1.png "Internet1"
[internet_image2]: ./examples/Internet_image2.png "Internet2"
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

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a learning rate of 0.001. Batch size of 128. I also tried using higher batch size such as 256, but the validation accuracy was coming lower using it. 128 worked better then batch size of 64 as well.
I finalized with epochs of 30. Tried with higher epochs but the accuracy was not increasing and was remaining the same or even decreasing a little.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The approach and important points for finalizing on the final architecture and training and validating the models are as follows.

- First of all normalizing the images turned out to be important for increasing the accuracy by 5 to 6 percent.
- Second application of dropout helped in mitigating the problem of overfitting and helped in increase in val accuracy by approx 2 percent
- Third from implementation perspective, had to take care that the validation script is made in such a way that during evaluation the dropout values become 1.0 because during evaluation the dropout feature should not be used.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.1%
* test set accuracy of 94.2%

I did not even try out with training using only grayscale images as while looking at the images it looked like colour could play an important role as a feature.
Also, after training and looking at training accuracy, it was clear that the even though the model is very small (only 2 convolutional layer) the train accuracy was easily going up to 99%. So, there was no need to simplify the features by converting to grayscale as the model was very well equiped to handle the current dataset size.

Tried with lower and higher learning rates like 0.01 and 0.0001 but the loss was not decreasing as required.

The validation accuracy of 96% and test accuracy of 94% is very high considering the number of classes are 43. So, the model is able to generalize well to the test data. However, for real world data it probably won't perform that well as in real world images, the variation in images could be much different that is not even specified in even training data. Data Augmentation would help in training and probably we would need to preprocess the real world images and convert them to how they look like in training images for the model to work with similar amount of accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![internet_image1][internet_image1]

![internet_image2][internet_image2]

First of all, it was difficult and time consuming task to find appropriate german traffic sign images on the Internet which were not copyrighted or did not contain a label or stamp over the image.

If you observe in the Jupyter notebook where I have also displayed the updated resolution (32X32) of the images, the "train" sign and "pedestrian and cyclist" sign are difficult or impossible to recognise even with our eyes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work  									| 
| Stop sign     			| Stop sign										|
| pedestrian					| Keep right											|
| train	      		| slippery road					 				|
| not sure			| Right-of-way at the next intersection      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. Not sure if the train image is in the train set or what the final image actually is. It was difficult to get hold of images from the Internet and I am not aware of German signs that much.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

| Probability         	|     Prediction top1				| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Road work   									| 
| .99     				| Stop sign 										|
| .998					| keep right											|
| .99	      			| slippery road					 				|
| 1.0				    | Right-of-way at the next intersection      							|

By looking at the softmax probablities the model is always confident even if the prediction is wrong or might be wrong. Thus, it needs more finetuning to handle natural occuring images. Probably data augmentaion techniques would help and also maybe some of the images that I am using are not correct or representative of the training set images. **Because of time consumption in finding correct images on the Internet and knowing about German traffic signs I am not wasting more time on it as I have understood the concept overall.**
