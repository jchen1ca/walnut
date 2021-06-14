Table of Contents:

- [Introduction](#intro)
- [Pros and Cons](#pros)
- [Data Augmentation Guidelines](#guidelines)
- [Common Image Augmentation Methods](#methods)
   - [Flips](#flips)
   - [Crops](#crops)
   - [Color Jitter](#jitter)
   - [Translation](#translation)
   - [Rotation ](#rotation)
   - [Stretching](#stretching)
   - [Shearing](#shearing)
   - [Lens Distortions](#distortion)
   - [Local Warping](#warp)
   - [Erasing](#erase)
   - [Contrast](#contrast)
   - [Gaussian Noise](#noise)
   - [Reinforcement](#reinforcement)
   - [Filter Kernel](#filter)
   - [Mix Images](#mix)
   - [Image Simulation](#simulation)
- [Situations not Applicable](#applicable)
- [Common image Augmentation Packages](#package)
- [Summary](#summary)

<a name='intro'></a>
## Introduction
Data augmentation, the technique of artificially expanding training dataset, is very popular in vision deep learning. It is used as a secret sauce in nearly every state-of-the-art model for image classification, and is becoming increasingly common in other modalities such as natural language understanding as well. Except practicing and examples, there are more and more [threories](http://ai.stanford.edu/blog/data-augmentation/) trying to explain how data augmentation works. 
[Dao et al. 2019](http://proceedings.mlr.press/v97/dao19b/dao19b.pdf) show that data augmentation model combined with a k-nearest neighbor (k-NN) classifier is asymptotically equivalent to a kernel classifier. It can be approximately decomposed into two components: (i) an averaged version of the transformed features, and (ii) a data-dependent variance regularization term.

<a name='pros'></a>
## Data Augmentation Pros and cons
There are advantages and disadvantages of Data augmentation for improving deep learning in image classification problem.\
**Pros:**
-	Induce invariance and regularization, reduce model complexity. 
-	Improve generalization. 
-	Improve robustness, reduce overfitting. 
-	Make a CNN model to be invariant to translation, viewpoint, size or illumination. [Link](https://openreview.net/forum?id=p84tly8c4zf)
-	Artificially increase training and testing dataset even you have a small dataset

**Cons:**
-	Too much of image augmentation combined with other forms of regularization (weight L2, dropout) can cause the net to underfit.
-	Too much image augmentation can lead to decreased accuracy in training and validation. 
-	Data augmentation can bring data bias, i.e. the augmented data distribution can be quite different from the original one. [Link]( https://arxiv.org/abs/2010.01267)

<a name='guidelines'></a>
## Data Augmentation Guidelines
-	Data augmentation shall increase in information and a better basis for decision making. 
-	Can focus on the feature and generalize it. [Link](https://arxiv.org/ftp/arxiv/papers/1901/1901.06032.pdf)
-	Avoid data basis which lead to divergence
-	As soon as the transformations can result to an image semantically consistent, i.e. you still can tell it is a cat. 
-	Also can combine these transformations, for example, translate combined with rotate, stretch and shear, 

<a name='methods'></a>
## Common Image Augmentation Methods
[Krizhevsky et al. 2012, ResNet](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf), used data augmentation to  reduce overfitting on image data. In this paper, the data augmentation consists of generating image translations and horizontal reflections. Now more and more data augmentation methods are introduced to image data deep learning. Below is a list of common image data augmentation methods.

<a name='flips'></a>
### 1)	Horizontal flips
Mirror each pixel with the vertical axis of the image. Horizontal flipping is widely used in almost all image augmentation. It is more popular than vertical flipping which sometimes is not appliable.
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/flip.jpg" width="50%">
  <div class="figcaption">
   
 [This image](https://www.flickr.com/photos/malfet/1428198050) by [Nikita](https://www.flickr.com/photos/malfet/) is licensed under [CC-BY2.0](https://creativecommons.org/licenses/by/2.0/)

<a name='crops'></a>
### 2)	Crops
A section of the image is sampled randomly. Normally followed by image resize or rescale.\
Training: sample random crops/scales\
Resnet:
   -	Cat is partially missed in each image
   -	Pick random L in range [256,480]
   -	Resize training image, short side = L
   -	Sample random 224x224 patch
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop1.jpg" width="25%">
  <div class="figcaption">

Testing: average a fixed set of crops\
ResNet:
   -	Resize image at 5 scales: {224,256,384,480,640}
   -	For each size, use 10 224x224 crops: (4 corners + 1 center)  x flips 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop2.jpg" width="50%">
  <div class="figcaption">
 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop3.jpg" width="50%">
  <div class="figcaption">
 
<a name='jitter'></a>
### 3)	Color jitter

   -	Randomize contrast and brightness
   -	Apply PCA to all R,G,B channels in training set
   -	Sample color offset along principal component directions
   -	Add grayscale offset to all pixels of a training image
   -	Hue jitter shifts the hue by a random amount
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/color_jitter.jpg" width="50%">
  <div class="figcaption"> 

<a name='translation'></a>
### 4)	Translation
Translation is to shift the image left or right, up or down, on a ratio that defines how much maximum to shift. To resize the image back to its original dimensions Keras by default uses a filling mode called ‘nearest’.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/translation.jpg" width="50%">
  <div class="figcaption">
   
<a name='rotation'></a>
### 5)	Rotation 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/rotation.jpg" width="50%">
  <div class="figcaption">
  
<a name='stretching'></a>
### 6)	Stretching
-	Contrast stretching 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/stretch.jpg" width="50%">
  <div class="figcaption">
   
<a name='shearing'></a>
### 7)	Shearing
-	To change rectangle image to parallelogram
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/shear.jpg" width="50%">
  <div class="figcaption">
   
<a name='distortion'></a>
### 8)	Lens distortions
-	In different viewpoint, lens distortion describe the object differently in scale and correlation
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/distortion.jpg" width="50%">
  <div class="figcaption">
     
[Sebastian Lutz, et al] (https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1001&context=impstwo)
   
<a name='warp'></a>
### 9)	Local warping
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/warp.jpg" width="50%">
  <div class="figcaption">
     
[Link1](https://arxiv.org/pdf/1609.08764.pdf)
[Link2](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14651)
   
<a name='erase'></a>
### 10)	Erasing
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/erise.jpg" width="50%">
  <div class="figcaption">
     
[Link]( https://ojs.aaai.org/index.php/AAAI/article/view/7000 )

<a name='contrast'></a>
### 11)	Contrast / histogram processing 
Change image contrast by maximize image histogram. The newly created images can be used to pre-train the given neural network in order to improve the training process efficiency. But data deficiency is one of the most relevant issues. 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/contrast.jpg" width="50%">
  <div class="figcaption">
     
See the link [Agnieszka M. et al, 2018](https://ieeexplore.ieee.org/abstract/document/8388338)
     
<a name='noise'></a>
### 12)	Blur image / add Gaussian noise
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/noise.jpg" width="50%">
  <div class="figcaption">

<a name='reinforcement'></a>
### 13)	Using reinforcement learning to do image data augmentation 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/reinforce.jpg" width="50%">
  <div class="figcaption">
Cubuk et al. AutoAugment: 
Learning Augmentation Strategies from Data, CVPR 2019

<a name='filter'></a>
### 14)	 Apply Filter kernel
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/filterKernel.jpg" width="50%">
  <div class="figcaption">
   
<a name='mix'></a>
### 15)	 Mix images
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/mixed.jpg" width="50%">
  <div class="figcaption">
Another example: Train on rando blends of images, i.e. 40% cat, 60% dog. Then use original images for test.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/blend.jpg" width="50%">
  <div class="figcaption">
   
<a name='simulation'></a>
### 16)	Image simulation
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/simulation.jpg" width="50%">
  <div class="figcaption">

<a name='applicable'></a>
## Situations that Data Augmentation is  not Applicable
1)	OCR can’t use vertical flipping, because “6” after flipping is “9”.
2)	

<a name='package'></a>
## Common image Augmentation Packages
-	keras.preprocessing.image.[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
-	[imaug](https://imgaug.readthedocs.io/en/latest/)
-	[albumentations](https://albumentations.ai/)
-	[opencv](https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5)
-	[augmentor](https://augmentor.readthedocs.io/en/master/)
-	[skimage](https://scikit-image.org/docs/dev/api/skimage.html)
-	[solt](https://mipt-oulu.github.io/solt/)
   
<a name='example'></a>
## Examples on how to use image augmentation
-	Procedures on how to choose augmentation methods
-	How to evaluation and test the augmentation
-	Analysis and comments

<a name='summary'></a>
## Summary


2)	Cubuk et al. AutoAugment:  Learning Augmentation Strategies from Data, CVPR 2019
3)	https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0#ref-CR6 
4)	https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/ 
5)	https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/ 
