# Homework 1

> 2018000337
> 컴퓨터소프트웨어학부
> 장호우

## 1-1

1. add `lr_utils.py` as following,

   ```python
   import numpy as np
   import h5py
       
       
   def load_dataset():
       train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
       train_set_x_orig = np.array(train_dataset["train_set_x"][:])
       train_set_y_orig = np.array(train_dataset["train_set_y"][:])
   
       test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
       test_set_x_orig = np.array(test_dataset["test_set_x"][:])
       test_set_y_orig = np.array(test_dataset["test_set_y"][:])
   
       classes = np.array(test_dataset["list_classes"][:])
       train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
       test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
       
       return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
   ```
   
2. In the 19th cell, modify `classes[d["Y_prediction_test"][0, index]].decode("utf-8")` to `classes[np.squeeze(test_set_y[:, index])].decode("utf-8")`

   ```python
    # Example of a picture that was wrongly classified.
    index = 5
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[np.squeeze(test_set_y[:, index])].decode("utf-8") +  "\" picture.")
   ```
   
3. In the 22th cell, the old version of `scipy.ndimage` does not support `imread`, import `PIL.Image` to load image

   ```python
   from PIL import Image
   ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
   my_image = "my_image.jpg"   # change this to the name of your image file 
   ## END CODE HERE ##
   
   # We preprocess the image to fit your algorithm.
   fname = "images/" + my_image
   image = np.array(Image.open(fname).resize((num_px, num_px)))
   my_image = np.reshape(image, (1, num_px * num_px * 3)).T
   my_predicted_image = predict(d["w"], d["b"], my_image)
   
   plt.imshow(image)
   print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
   ```

Screenshot of running result

<img src="E:\Download\screencapture-localhost-8888-lab-tree-Desktop-ITC4009-deep-learning-coursera-Neural-Networks-and-Deep-Learning-Logistic-Regression-with-a-Neural-Network-mindset-ipynb-2021-03-25-21_35_19.png" style="zoom: 50%;" />

## 1-2

Cell 1: import the libraries in need

Cell 2: loading train and test data

Cell 3: show a example (26th, idx=25) image of train dataset

Cell 4: print the shape of images in train/test dataset

Cell 5: reshape the RGB images, flatten them to row vectors

The train/test dataset is in four dimensions (as Cell 4), should be reduce the dimension,

make (a, b, c, d) to (b\*c\*d,  a)

```python
# reduce the dimension of the training/testing set and transpose it
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
```

Cell 6: standardize images of train/test dataset

Cell 7: build a sigmoid activation function

Cell 8: print results of sigmod(0.5) and sigmod(9.2) to verify that

Cell 9: initializing parameters(w) as zero vector

Cell 10: print the initialized vector that filled by default value, zero

Cell 11: build a propagation function via loss function of logistic regression

```python
# size of  train data
m = X.shape[1]
# activation via sigmoid function
# do matrix multiplication with the transpose of w(1, 64 * 64 * 3) and X (12288, 209)
A = sigmoid(np.dot(w.T, X) + b)
# compute cost of forward propagation
cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
# backward propagation
dw = (1 / m) * np.dot(X, (A - Y).T)
db = (1 / m) * np.sum(A - Y)
```

Cell 12: print the results of propagation, cost and gradient of the loss



