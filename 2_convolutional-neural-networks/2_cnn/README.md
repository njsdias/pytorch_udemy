# Files to work
- conv-visualization/custom_filters.ipynb : applies different filter an image
- conv-visualization/maxpooling_visualization.ipynb: shows the effect of the maxpooling layer in an image
- conv-visualization/conv_visualization.ipynb: shows how to build a
CNN in Pytorch and shows the effect of the ReLu activation function
- cifar-cnn/cifar10_cnn.ipynb: build a CNN's to classify images
- cifar-cnn/cifar10_cnn_augumentation.ipynb: shows how we can introduce scale, rotation and translation invariance in a model to classify images


# Convolution Neural Network

To use MLP (Multi Layer Perceptron) to image classification we have some issues:
- MLP's uses a lot of  parameters that is a huge problem for problems
that involves images with more complex features than MNIST dataset which is a dataset where images are much well pre-processed.
- Do not consider the 2D information contained in the matrix that represents the image when we flat this matrix image into a

The CNN is a solution for these problems using layers that are sparsely connected where the connections between layers are informed by the 2D structure of the image matrix. Besides of that, CNN accept matrix as input. Remember, MLP's only accepts vectors as input.

In  the CNN's the neural network is building by split an image in several sections. The pixels in each section are connected with each other in these first hidden layer. In other words, the pixels that belongs of a section are not connect to pixels that belongs o other section. It reduces these number of connections and the parameters. Increasing the number of nodes in hidden layer is possible to capture more complex pattern from each image section. (https://www.youtube.com/watch?v=z9wiDg0w-Dc)  

A CNN can:
- remember spatial information.
- look an image as a whole or in patches and analyze groups of pixels at a time

A CNN applies different images filters also know as convolutional kernels to an input image. The resulting filtered images have different appearances (color/shape).

Shape can be thought of as patterns of intensity in an image. Intensity is a measure of light and dark (brightness) in an image, that are used to detect the shape of objects in an image.

## **Frequency in an Image**
A high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly. In general, images have high and low intensities.

**High-frequency components also correspond to the edges of objects in images**, which can help us classify those objects.

## **Filters in images**
In images processing, filters are used to filter out unwanted or irrelevant information in an image or to amplify features like object boundaries or other distinguishing traits.

**High-pass filters** are used to make an image appear sharper and enhance high-frequency parts of an image, which are areas where the levels of intensity in neighboring pixels rapidly change like from very dark to very light pixels.

These filters are used to emphasize edges in a gray scale image.

## Convolution Kernels
The filters are in the form of matrices often called convolution kernels.
A kernel is a matrix of numbers that modifies an image-

*Important note*: Kernel convolution is represented by an asterisk (it is not the multiply symbol) and it is an important  operation in computer vision.

The kernel (matrices with sum all elements equal to zero) is apply in all picture parts to change the values of the pixel to highlight for example, the object edges.

See the video below:
- https://www.youtube.com/watch?time_continue=148&v=OpcFn_H2V-Q&feature=emb_logo

We want to chose the best filter for finding and enhancing horizontal edges and lines in an image? Remember that these filters calculate a difference between neighboring pixels and around a center pixel. Which difference would best detect horizontal lines?

This kernel finds the difference between the top and bottom edges surrounding a given pixel.

|-1|-2|-3|
|---|---|---|
|**0**|**0**|**0**|
|**1**|**2**|**3**|

Exercise file:
2_cnn/convolutional-neural-networks/conv-visualization/**custom_filters.ipynb**

## The Importance of Filters
 The CNN's uses **convolution layers** keep track of spatial information and learn to extract features like the edges of objects.

 The convolution layers is compounded by several filters known as convolution kernels. If we use for instance four filters (with each filter you obtain a new modified image), and stack these images,  we form a form a complete **convolutional layer with a depth of 4**.

 In the previous exercise, the filters was coded from scratch. In practice, many neural networks learn to detect the edges of images because the edges of object contain valuable information about the shape of an object.

## Additional notes for cifar10_cnn

The input image is 32x32x3. The number 3 means convolution kernel of size 3 by 3 When we do:

`nn.Conv2d(3, 16, 3, padding=1)`

we are saying that we are starting with an image depth of 3, moving it to 16, maintain constant the size of convolution kernel. **Image depth** is the number of layers presented in an image. The colored images has 3 layers (Red-Green-Blue). If we're working with gray images, it meas the depth of the image is One

The `nn.Conv2d(16, 32, 3, padding=1)` means we moving the image depth from 16 to 32.

The `nn.Conv2d(32, 64, 3, padding=1)` means we moving the image depth from 32 to 64.

The `nn.MaxPool2d(2, 2)` down-sample any xy size by two. It will be applied in each Conv2d layer.

The `nn.Linear(64 * 4 * 4, 500)` means, we take the last output (64) and multiply it by 4. We obtain 4 number doing 32/2/2/2. It means we started with a image of 32x32 size. Because we apply the MAxPool2d in each Conv2d, that means down-sample any xy size by two, we need to divide by 2 in each Conv2d. Because we have three Conv2d, we need divide three times.

The `nn.Linear(500, 10)` means we start from the output (500) of last layer and goes to 10 that represents the number of classes.

## Pooling layers
These layers prevents the over-fitting and reduces the dimensionality of our problem. Imagine we have a complicate dataset with several categories. We need many filters in each is responsible to find a pattern in the image. These filters are responsible to find more complex patterns increasing the number of parameters (model with high dimensional) which can lead to over-fitting.

**Max Pooling layers** receives as input *windows size* and *stride*.

See the video: https://www.youtube.com/watch?v=_Ok5xZwOtrk

It's important to note that pooling operations do throw away some image information. That is, they discard pixel information in order to get a smaller, feature-level representation of an image. This works quite well in tasks like image classification, but it can cause some issues (face recognition with these technique can generated outputs with non a good results).

To solve this problem there is a technique names as **capsule network** which learns spatial relationships between parts. This means that capsule networks are able to recognize the same object, like a face, in a variety of different poses and with the typical number of features (eyes, nose , mouth) even if they have not seen that pose in training data.

See more about capsule network: https://cezannec.github.io/Capsule_Networks/


# **Convolutional Layers in PyTorch**
To create a convolutional layer in PyTorch, you must first import the necessary module:

`import torch.nn as nn`

Then, there is a two part process to defining a convolutional layer and defining the feedforward behavior of a model (how an input moves through the layers of a network). First, you must define a Model class and fill in two functions.

**init**

You can define a convolutional layer in the `__init__` function of by using the following format:

`self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`

**forward**

Then, you refer to that layer in the forward function! Here, I am passing in an input image `x` and applying a ReLU function to the output of this layer.

`x = F.relu(self.conv1(x))`

**Arguments**

You must pass the following arguments:

**in_channels** - The number of inputs (in depth), 3 for an RGB image, for example.
**out_channels** - The number of output channels, i.e. the number of filtered "images" a convolutional layer is made of or the number of unique, convolutional kernels that will be applied to an input.
**kernel_size** - Number specifying both the height and width of the (square) convolutional kernel.

There are some additional, optional arguments that you might like to tune:

**stride** - The stride of the convolution. If you don't specify anything, stride is set to 1.
**padding** - The border of 0's around an input array. If you don't specify anything, padding is set to 0.

*NOTE*: It is possible to represent both kernel_size and stride as either a number or a tuple.

There are more tunable arguments. You can find more here:
https://pytorch.org/docs/stable/nn.html#conv2d


**Pooling Layers**
Pooling layers take in a kernel_size and a stride. Typically the same value as is the down-sampling factor. For example, the following code will down-sample an input's x-y dimensions, by a factor of 2:

`self.pool = nn.MaxPool2d(2,2)`

**forward**

Here, we see that poling layer being applied in the forward function.
```
x = F.relu(self.conv1(x))
x = self.pool(x)
```

**Convolutional Example #1**
Say I'm constructing a CNN, and my input layer accepts grayscale images that are 200 by 200 pixels (corresponding to a 3D array with height 200, width 200, and depth 1). Then, say I'd like the next layer to be a convolutional layer with **16 filters**, each filter having a **width and height of 2**. When performing the convolution, I'd like the filter to **jump two pixels at a time**. I also don't want the filter to extend outside of the image boundaries; in other words, I don't want to pad the image with zeros. Then, to construct this convolutional layer, I would use the following line of code:

`self.conv1 = nn.Conv2d(1, 16, 2, stride=2)`

**Convolutional Example #2**
Say I'd like the next layer in my CNN to be a convolutional layer that **takes the layer constructed in Example 1 as input (16)**. Say I'd like my new layer to have **32 filters**, each with a **height and width of 3**. When performing the convolution, I'd like the **filter to jump 1 pixel at a time**. I want **this layer to have the same width and height as the input layer, and so I will pad accordingly**. Then, to construct this convolutional layer, I would use the following line of code:

Note: By default stride=1. So If I want jump one pixel at the time I don't need to specify stride=1

`self.conv2 = nn.Conv2d(16, 32, 3, padding=1)`

Convolution with 3x3 window and stride 1
https://iamaaditya.github.io/2016/03/one-by-one-convolution/

Image source: http://iamaaditya.github.io/2016/03/one-by-one-convolution/

**Sequential Models**
We can also create a CNN in PyTorch by using a Sequential wrapper in the `__init__` function. Sequential allows us to stack different types of layers, specifying activation functions in between!
```
def __init__(self):
        super(ModelName, self).__init__()
        self.features = nn.Sequential(
              nn.Conv2d(1, 16, 2, stride=2),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True),

              nn.Conv2d(16, 32, 3, padding=1),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True)
         )
```

**Formula: Number of Parameters in a Convolutional Layer**
The number of parameters in a convolutional layer depends on the supplied values of filters/out_channels, kernel_size, and input_shape. Let's define a few variables:

- K - the number of filters in the convolutional layer
- F - the height and width of the convolutional filters
- D_in - the depth of the previous layer

Notice that `K = out_channels`, and `F = kernel_size`. Likewise, `D_in` is the last value in the input_shape tuple, typically 1 or 3 (RGB and grayscale, respectively).

Since there are `F*F*D_in` weights per filter, and the convolutional layer is composed of `K` filters, the total number of weights in the convolutional layer is `K*F*F*D_in`. Since there is one bias term per filter, the convolutional layer has K biases. Thus, **the number of parameters in the convolutional layer** is given by `K*F*F*D_in + K`.

**Formula: Shape of a Convolutional Layer**
The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

- K: the number of filters in the convolutional layer
- S - the stride of the convolution
- P - the padding
- W_in - the width/height (square) of the previous layer

Notice that `K = out_channels`, `F = kernel_size`, and `S = stride`. Likewise, `W_in` is the first and second value of the input_shape tuple.

The depth of the convolutional layer will always equal the number of filters `K`.

The **spatial dimensions of a convolutional layer** can be calculated as: `(W_in−F+2P)/S+1`

**Flattening**
Part of completing a CNN architecture, is to flatten the eventual output of a series of convolutional and pooling layers, so that all parameters can be seen (as a vector) by a linear classification layer. At this step, it is imperative that you **know exactly how many parameters are output by a layer**.

# **Nomenclature**
- **Edge Handling**
Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use **padding, cropping, or extension**. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

- **Extend** The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

- **Padding** The image is padded with a border of 0's, black pixels.

- **Crop** Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

- **stride**: is just the amount by which the filter slides over the image. A stride of one makes the convolution layer roughly the same width and height as the input image. If we stride equal to two the convolution layer is about half the width and height of the image.

- **Maxpooling**: A maxpooling layer reduces the x-y size of an input and only keeps the most active pixel values. Maxpooling layers commonly come after convolutional layers to shrink the x-y dimensions of an input, read more about pooling layers in PyTorch.

- **A ReLu** This activation function turns all negative pixel values in 0's (black).

- **Image depth** is the number of layers presented in an image. The colored images has 3 layers (Red-Green-Blue). If we're working with gray images, it meas the depth of the image is One. (https://www.youtube.com/watch?v=YKif1KNpWeE)

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`

**in_channels** refers to the depth of an input. For a grayscale image, this depth = 1
out_channels refers to the desired depth of
 **the output**, or the number of filtered images you want to get as output
**kernel_size** is the size of your convolutional kernel (most commonly 3 for a 3x3 kernel)
**stride** and **padding** have default values, but should be set depending on how large you want your output to be in the spatial dimensions x, y

**scale, rotation, translation invariante** we need that our model has capabilities to classify images without taking scale (size) or rotation (angles) and translation (move in the next directions up down right left ) effects.

## Additional Notes
- The more convolutional layers you include, the more complex patterns in color and shape a model can detect. It's suggested that your final model include 2 or 3 convolutional layers as well as linear layers + dropout in between to avoid over-fitting.

## Research Site
- Stanford's CNN course: https://cs231n.github.io/convolutional-networks/#layers

- Networks that wons ImageNEt challenge to classify 1000 different categories- AlexNet, VGG16, , ResNet152



## QUIZ
**1**
How might you define a Maxpooling layer, such that it down-samples an input by a factor of 4? (A checkbox indicates that you should select ALL answers that apply.)

Answer:
`nn.MaxPoll2d(2,4)`
`nn.MaxPool2d(4,4)`
The best choice would be to use a kernel and stride of 4, so that the maxpooling function sees every input pixel once, but any layer with a stride of 4 will down-sample an input by that factor.

**2**
If you want to define a convolutional layer that is the same x-y size as an input array, what padding should you have for a kernel_size of 7? (You may assume that other parameters are left as their default values.)

Answer: padding=7

**3**
For the following quiz questions, consider an input image that is 130x130 (x, y) and 3 in depth (RGB). Say, this image goes through the following layers in order:
```
nn.Conv2d(3, 10, 3)
nn.MaxPool2d(4, 4)
nn.Conv2d(10, 20, 5, padding=2)
nn.MaxPool2d(2, 2)
```
1-
After going through all four of these layers in sequence, what is the depth of the final output?

Answer:The final depth is determined by the last convolutional layer, which has a depth = out_channels = 20

2-
What is the x-y size of the output of the final maxpooling layer? Careful to look at how the 130x130 image passes through (and shrinks) as it moved through each convolutional and pooling layer.

Answer:The 130x130 image shrinks by one after the first convolutional layer, then is down-sampled by 4 then 2 after each successive maxpooling layer!
130/4/2 = 16

3-
How many parameters, total, will be left after an image passes through all four of the above layers in sequence?

Answer: It's the x-y size of the final output times the number of final channels/depth = 16*16*20.
