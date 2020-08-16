# Fashion MNIST Image Classifier 
## Introduction
To start my journey on learning about neural networks using tensorflow on python, I have chosen to use a slightly different data set. This data set is called the fashion mnist(or fmnist), and like the conventional digits mnist data set, it contains 10 different labels. The labels ranges from values 0 to 9, where each digit is a representation of a type of clothing. This representation can also be seen in the table below 

| Label | Clothing Type | 
|-------|---------------|
|0 | T-shirt|
|1| Pants|
|2| Pullover Sweater|
|3 |Dress |
|4 | Coat|
|5 | Sandal|
|6 | Shirt|
|7 | Shoes|
|8 | Bag|
|9 | Boots|

# Packages used 

  * Tensorflow/keras
  * Numpy
  * Matplotlib

## Data Processing 

As this is a commonly used data set, the data requires very little cleaning. However, the features still need to be normalized; as such, I made use of tensorflow keras' `normalization` function. This function will process the features such that they will be between values of 0 and 1.

## Model 
For this particular project,  I though this would be an excellent opportunity to see just how well CNN can perform. Thus, I decided to compare and contrast the results from 2 different models. The first model is just your normal neural network model with simple dense layers. On the other hand, the second model contains convolution layers, max pooling layers and simple dense layers.


## Results
Taking a look below, we can see 2 graphs showing the accuracy & loss of both models on the training set. As expected with the CNN model, it was able to quickly achieve very high accuracies compared to the 'basic' model with just dense layers. In addition to the high accruacy, the CNN model was also able to quickly decrease its loss. 

<img src="imgs/accuracy plot.png"  width = 500/>
<img src="imgs/loss plot.png"  width = 500/>

Now taking a look at the models' performance on the test set, we a very similar result to the training set 

* **NN with simple dense layers:** 
  * Accuracy = 80%

* **Convolution NN:** 
  * Accuracy = 89%

