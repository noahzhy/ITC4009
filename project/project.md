# Image classification in embedded device

## Abstract

This project mainly discusses related problem of image classification in embedded device through some simple tasks to discuss them. It includes three parts:

1. The effect of using only one channel of RGB channels to replace the grayscale image as input.

2. The representative mobile network shuffleNet V2 and it's improvements.

3. The latency of some representative blocks in embedded device.

## Introduction

The traditional neural network, mAP, FLOPs and other evaluation indicators are not suitable for mobile devices or embedded devices in the actual project deployment. Mobile devices or embedded devices have limited clock rate and memory, and come with accelerated hardware for tasks in deep learning. These limitations are often overlooked, thus this project focuses on the deployment of some neural networks on mobile devices or embedded devices.

In the first topic, because of memory limitations in many embedded devices. If using RGB channel, it will open at least (width x height x channels x type of data) memory space. In addition, the clock rate of many embedded devices is around 200 MHz. Even converting an RGB image to a grayscale image is a huge drain on computing power (detail in Appendix 1). Therefore, if one channel in RGB channels can be directly used for the input of neural network via a memory pointer, it will greatly improve the efficiency of embedded device without the step of RGB to grayscale in deep learning. This topic discusses the effect of a single channel on recognition accuracy.

In the second topic, it is mainly an improvement of shuffleNet V2. In recognition or detection tasks for embedded devices, the relatively distance between the device and the object being detected is generally fixed. The traditional object detection and recognition networks usually consider the complex situation of small object or overlapping object, which is not suitable for the deep learning task of embedded devices.

The embedded devices designed for deep learning is generally consist of a low clock rate CPU with a hardware acceleration devices such as KPU, TPU, etc.. Third topic is mainly discuss the latency and efficiency of representative blocks on different acceleration hardware platforms.

## Dataset

The dataset in this project is `cifar10`.
* https://www.tensorflow.org/datasets/catalog/cifar10

## Methodology

In topic one, the same neural network(LeNet) was used but change the channel of inputs to test the effect of one single channel on the accuracy of image recognition.

LeNet via Keras

```python
def LeNet():
    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(32,32,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model
```

Modified the generator of dataset, thus we could choose generate only one channel in RGB images or convert to grayscale images.

```python
def dataset_generator(dataset, batch_size=256, num_classes=10, is_training=False, color_mode="grayscale"):
  images = np.zeros((batch_size, 32, 32, 1))
  labels = np.zeros((batch_size, num_classes))
  while True:
    count = 0 
    for sample in tfds.as_numpy(dataset):
      image = sample["image"]
      label = sample["label"]

      if color_mode == "grayscale":
        # rgb to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
      else: # keep only one channel in RGB
        channel_index = list('BGR').index(color_mode.upper())
        image = image[:,:,channel_index]

      images[count%batch_size] = mobilenet_v2.preprocess_input(np.expand_dims(cv.resize(image, (32, 32)), -1))
      labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)

      count += 1
      if (count%batch_size == 0):
        yield images, labels
```

在主题二中，通过对比原始shuffleNet以及改进后的变体的识别精度来测试性能的提升。

在主题三种，使用相同的block，通过更换不同的硬件平台以测试不同设备对于同一种block的性能的影响。

## Results

## Conclusion

## Appendix

### Appendix 1

The common code sample. Traversed each pixel of all channels and calculate the grayscale value.

```c++
typedef struct{
    int w, h, c;
    float *data;
} image;

image make_image(int w, int h, int c) {
    image out;
    out.h = h;
    out.w = w;
    out.c = c;
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

float get_pixel(image im, int x, int y, int c) {
    int index = ((im.h * im.w) * c) + (im.w * y) + x;
    return *(im.data + index);
}

void set_pixel(image im, int x, int y, int c, float v) {
    int index = ((im.h * im.w) * c) + (im.w * y) + x;
    *(im.data + index) = v;
    return;
}

image rgb_to_grayscale(image im) {
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    int i, j;
    for(i = 0; i < im.w; i++) {
        for(j = 0; j < im.h; j++) {
            float im_red = get_pixel(im, i, j, 0);
            float im_green = get_pixel(im, i, j, 1);
            float im_blue = get_pixel(im, i, j, 2);
            float grey_pixel = (0.299 * im_red) + (0.587 * im_green) + (0.114 * im_blue);
            set_pixel(gray, i, j, 0, grey_pixel);
        }
    }
    return gray;
}
```
