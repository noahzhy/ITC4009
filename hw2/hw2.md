# Homework 2

> 2018000337
> 컴퓨터소프트웨어학부
> 장호우

## 2-1-1

Screenshot of running result

<img src="../images/0001.png" style="zoom: 50%;" />

## 2-1-2

Cell 13: build the optimize function, implement a function that minimizes cost by repeating the propagation function implemented above(in hw1). This function returns the parameters of the optimization, calculates the value of gradient descent and cost

Cell 14: show the learned parameters

Cell 15: build the predict function

Cell 16: show the results of prediction

Cell 17: merge these functions above into the model function, initialize the parameter(w and b) with zeros via initialize_with_zeros(), learning the parameters via optimize, last, predict the images of dateset and print the accuracy of prediction

Cell 18: training the model

Cell 19: show the 6th image of test dataset and predicted result

Cell 20: Plot the cost value for each iteration and the cost go down after each iteration

Cell 21: set the learning rate to other values(0.01, 0.001 and 0.0001) for learning, and print each result. The accuracy get worse and worse in results. Thus, if the learning rate is too large (e.g. 0.01), then the cost will fluctuate significantly and even change. A small learning rate will result overfitting. If the learning rate is too small, it is okay for the training set, but will reduce the accuracy of the test set because of overfitting.

Cell 22: show the image and predicted result

## 2-2

Cell 1: import libraries what we need in this case

Cell 2: load the dataset and show the shape(size) of this dataset

Cell 3: visualize the dataset, the blue point is when y=1, red means t=0

Cell 4: check the amount of X, Y in this dataset

Cell 5: divide two areas via Logistic Regression in sklearn kit

Cell 6: show the results that using Logistic Regression, the accuracy only 47% in this method

Cell 7: the size of feature by shape(size) of input data, also is the size of the layer

Cell 8: print the shape of each layer(input layer, hidden layer, output layer)

Cell 9: initialize parameters, in neural network, must initialize w to a non-zero random value. Otherwise, it doesn't work, but parameter b can be initialized to zero. Initialize it via numpy

Cell 10: show the result that after initializing parameters randomly

Cell 11: build the forward propagation function, using tanh activation function in hidden layer, using the sigmoid activation function in output layer

Cell 12: check output

Cell 13: calculate the value of cost via the formula between cell 12 and cell 13

Cell 14: check output

Cell 15: build a backward propagation function, calculate the differential, and update the each parameter

Cell 16: check output

Cell 17: If the learning rate is appropriate, the algorithm will look for parameters that work well and minimize the cost, but if the learning rate is too high, it won't work

Cell 18: check output(each parameter)

Cell 19: build neural network model, initializes the parameters, execute forwad_propagation(), compute_cost() and backward_propagation() each iteration, and updates the parameters with gradient value

Cell 20: check output(each parameter)

Cell 21: predict the class by learned parameters, as before, if greater than 0.5 is 1, less than 0.5 is 0

Cell 22: check results of prediction

Cell 23: set the size of X, Y data and hidden layer is 4 to nn_model, learning results as following

Cell 24: the eparation of Decision Boundary is very accurately and accuracy up to 90%  
