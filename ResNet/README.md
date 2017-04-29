# RestNet (ILSVRC 2015) #

Paper: Deep Residual Learning for Image Recognition 
by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
https://arxiv.org/abs/1512.03385

![alt tag](https://github.com/rainer85ah/Papers2Code/blob/master/ResNet/resnet.png)

ResNet have a simple ideas: feed the output of two successive convolutional layer AND also bypass the input to the next layers!

![alt tag](https://github.com/rainer85ah/Papers2Code/blob/master/ResNet/resnetbottleneck.jpg)

Here they bypass TWO layers and are applied to large scales. Bypassing after 2 layers is a key intuition, as bypassing a single layer did not give much improvements. By 2 layers can be thought as a small classifier, or a Network-In-Network!

This is also the very first time that a network of > hundred, even 1000 layers was trained.

ResNet with a large number of layers started to use a bottleneck layer similar to the Inception bottleneck:

![alt tag](https://github.com/rainer85ah/Papers2Code/blob/master/ResNet/resnetb.jpg)

This layer reduces the number of features at each layer by first using a 1x1 convolution with a smaller output (usually 1/4 of the input), and then a 3x3 layer, and then again a 1x1 convolution to a larger number of features. Like in the case of Inception modules, this allows to keep the computation low, while providing rich combination of features. See “bottleneck layer” section after “GoogLeNet and Inception”.

Additional insights about the ResNet architecture are appearing every day:

* ResNet can be seen as both parallel and serial modules, by just thinking of the inout as going to many modules in parallel, while the output of each modules connect in series.

* ResNet can also be thought as multiple ensembles of parallel or serial modules.

* It has been found that ResNet usually operates on blocks of relatively low depth ~20-30 layers, which act in parallel, rather than serially flow the entire length of the network.

* ResNet, when the output is fed back to the input, as in RNN, the network can be seen as a better bio-plausible model of the cortex
