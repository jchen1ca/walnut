Table of Contents:

- [Introduction](#intro)
- [Image Data Augmentation Pros and Cons](#pros)
- [Image Data Augmentation Guidelines](#guidelines)
- [Common Image Data Augmentation Methods](#methods)
   - [Flipping](#flips)
   - [Cropping](#crops)
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
   - [GAN Based](#gan)
- [Situations not Applicable](#applicable)
- [Common image Data Augmentation Packages](#package)

<a name='intro'></a>
## Introduction
Data augmentation, the technique of artificially expanding training dataset, is very popular in vision deep learning. It is used as a secret sauce in nearly every state-of-the-art model for image classification, and is becoming increasingly common in other modalities such as natural language understanding as well[[Sharon Y. Li 2020]](http://ai.stanford.edu/blog/data-augmentation/). Except practicing and examples, there are more and more threories trying to explain how data augmentation works. [Dao et al. 2019](http://proceedings.mlr.press/v97/dao19b/dao19b.pdf) show that data augmentation model combined with a k-nearest neighbor (k-NN) classifier is asymptotically equivalent to a kernel classifier. It can be approximately decomposed into two components: (i) an averaged version of the transformed features, and (ii) a data-dependent variance regularization term.

<a name='pros'></a>
## Image Data Augmentation Pros and cons
There are advantages and disadvantages of Data augmentation for improving deep learning in image classification problem.\
**Pros:**
-	Induce invariance and regularization, reduce model complexity. 
-	Improve generalization. 
-	Improve robustness, reduce overfitting. 
-	Make a CNN model to be invariant to translation, viewpoint, size or illumination. [[Yi Xu, et al, 2021]](https://openreview.net/forum?id=p84tly8c4zf)
-	Artificially increase training and testing dataset even you have a small dataset

**Cons:**
-	Too much of image augmentation combined with other forms of regularization (weight L2, dropout) can cause the net to underfit.
-	Too much image augmentation can lead to decreased accuracy in training and validation. 
-	Data augmentation can bring data bias, i.e. the augmented data distribution can be quite different from the original one. [[Yi Xu, et al, 2020]]( https://arxiv.org/pdf/2010.01267.pdf)

<a name='guidelines'></a>
## Image Data Augmentation Guidelines
-	Data augmentation shall increase in information and a better basis for decision making. 
-	Can focus on the feature and generalize it. [[Asifullah Khan, et al, 2020]](https://arxiv.org/ftp/arxiv/papers/1901/1901.06032.pdf)
-	Avoid data basis which lead to divergence
-	As soon as the transformations can result to an image semantically consistent, i.e. you still can tell it is a cat. 
-	Also can combine these transformations, for example, translate combined with rotate, stretch and shear, 

<a name='methods'></a>
## Common Image Data Augmentation Methods
[Krizhevsky et al. 2012, ResNet](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf), used data augmentation to  reduce overfitting on image data. In that paper, the data augmentation consists of generating image translations and horizontal reflections, and proves to reduce overfitting and increase prediction accuracy. Generally, we can classify image data augmentation into two main groups: **position augmentation**, which includes scaling, cropping, flipping, padding, rotation, translation, affine transformation; and **color augmentation** which includes brightness, contrast, saturation, hue[Harshit Kumar, 2021](https://iq.opengenus.org/data-augmentation/). New data augmentation methods are introduced to image data deep learning, such as mixing, simulation, GAN based, etc. Below is a list of common image data augmentation methods.

<a name='flips'></a>
### 1)	Horizontal flipping
This example is used in CS231N lecture 8. Horizontal flipping is widely used in almost all image augmentation. It is more popular than vertical flipping which sometimes is not appliable. Horizontal flips is also called mirror image horizontally. 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/flip.jpg" width="40%">
  <div class="figcaption">
   
 [This image](https://www.flickr.com/photos/malfet/1428198050) by [Nikita](https://www.flickr.com/photos/malfet/) is licensed under [CC-BY2.0](https://creativecommons.org/licenses/by/2.0/)

<a name='crops'></a>
### 2)	Cropping
A section of the image is sampled randomly. Normally followed by image resize or rescale. Crop partial of the original image for training can force the model to find the cat even it's partially present, for example, see its tail only.\
Training: sample random crops/scales\
Resnet:
   -	Cat is partially missed in each image
   -	Pick random L in range [256,480]
   -	Resize training image, short side = L
   -	Sample random 224x224 patch
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop1.jpg" width="20%">
  <div class="figcaption">

Testing: average a fixed set of crops\
ResNet:
   -	Resize image at 5 scales: {224,256,384,480,640}
   -	For each size, use 10 224x224 crops: (4 corners + 1 center)  x flips 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop2.jpg" width="50%">
  <div class="figcaption">
  <br/>
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
Translation is to shift the image left or right, up or down, on a ratio that defines how much maximum to shift. To resize the image back to its original dimensions Keras by default uses a filling mode called ‘nearest’. In the following example, we assume that the image has a black background beyond its boundary, and are translated appropriately. This method of augmentation is very useful as most objects can be located at almost anywhere in the image. This forces your convolutional neural network to look everywhere. [Arun Gandhi, 2021](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/translation.jpg" width="50%">
  <div class="figcaption">
   
<a name='rotation'></a>
### 5)	Rotation 
Rotation can provide the cases of different orientation so model can learn to look for the object in various possibility. Rotation is a nasty data augmentation due to the blank border after rotating an angle not 90 or 180 degree. [Arun Gandhi, 2021](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/rotation.jpg" width="50%">
  <div class="figcaption">
  
     
<a name='stretching'></a>
### 6)	Stretching
The contrast stretching is a tool to normalize or narrow image contrast. This provides model with the chance to detect objects in blur or foggy situation.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/stretch.jpg" width="50%">
  <div class="figcaption">
   
<a name='shearing'></a>
### 7)	Shearing
Shearing is a bounding box transformation that can be done with the help of the transformation matrix. In shearing, we turn the rectangular image into a parallelogrammed image. [Ayoosh Kathuria, 2018](https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/). Shearing can describe the situation when the camera is in a skewed view angle.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/shear.jpg" width="50%">
  <div class="figcaption">
   
<a name='distortion'></a>
### 8)	Lens distortions
In different viewpoint, lens distortion describe the object differently in scale and correlation. Understanding the impact of lens distortion on deep learning can help us to use image augmentation to remediate the influence.  [Sebastian Lutz, et al, 2019](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1001&context=impstwo)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/distortion.jpg" width="50%">
  <div class="figcaption">
    
   
<a name='warp'></a>
### 9)	Local warping
This paper uses local warping to create more smaples when training a machine learning classifier. New data is generated through transformations applied in the data-space. This provides a great benefit for imporving performance and reducing overfitting.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/warp.jpg" width="50%">
  <div class="figcaption">
     
[Sebastien C. Wong, et al, 2016](https://arxiv.org/pdf/1609.08764.pdf)
[Hong Liu, et al, 2020](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14651)
   
<a name='erase'></a>
### 10)	Erasing
This paper introduces random erasing data augmentation method for training the convolutional neural network (CNN). In training, random erasing randomly selects a rectangle region in an image and erases its pixels with random values. In this process, training images with various levels of occlusion are generated, which reduces the risk of over-fitting and makes the model robust to occlusion. Random Erasing is parameter learning free, easy to implement, and can be integrated with most of the CNN-based recognition models. [Zhun Zhong, et al, 2019]( https://ojs.aaai.org/index.php/AAAI/article/view/7000 )
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/erise.jpg" width="50%">
  <div class="figcaption">
     

<a name='contrast'></a>
### 11)	Contrast / histogram processing 
Change image contrast by maximize image histogram. The newly created images can be used to pre-train the given neural network in order to improve the training process efficiency. But data deficiency is one of the most relevant issues. [Agnieszka M. et al, 2018](https://ieeexplore.ieee.org/abstract/document/8388338)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/contrast.jpg" width="50%">
  <div class="figcaption">
    
     
<a name='noise'></a>
### 12)	Add Gaussian noise
Over-fitting usually happens when your neural network tries to learn high frequency features (patterns that occur a lot) that may not be useful. Gaussian noise, which has zero mean, essentially has data points in all frequencies, effectively distorting the high frequency features. This also means that lower frequency components (usually, your intended data) are also distorted, but your neural network can learn to look past that. Adding just the right amount of noise can enhance the learning capability. [Arun Gandhi, 2021](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/noise.jpg" width="50%">
  <div class="figcaption">

     
<a name='reinforcement'></a>
### 13)	Using reinforcement learning to for image data augmentation 
This paper describes a procedure called AutoAugment to automatically search for improved dataaugmentation policies. And use the search algorithm to find the best policy such that the neural network yields the highest validation accuracy on a target dataset. [Cubuk et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/reinforce.jpg" width="50%">
  <div class="figcaption">


<a name='filter'></a>
### 14)	 Apply Filter kernel
Kernel filters are a very popular technique in image processing to sharpen and blur images. These filters work by sliding an n × n matrix across an image with either a Gaussian blur filter, which will result in a blurrier image, or a high contrast vertical or horizontal edge filter which will result in a sharper image along edges. Intuitively, blurring images for Data Augmentation could lead to higher resistance to motion blur during testing. Additionally, sharpening images for Data Augmentation could result in encapsulating more details about objects of interest. [Guoliang K, et al. 2017](https://arxiv.org/pdf/1707.07103.pdf)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/filterKernel.jpg" width="50%">
  <div class="figcaption">
   
     
<a name='mix'></a>
### 15)	 Mix images
The concept of mixing images in an unintuitive way was further investigated by [Summers and Dinneen](https://arxiv.org/pdf/1805.11272.pdf). They looked at using non-linear methods to combine images into new training instances. All of the methods they used resulted in better performance compared to the baseline models. 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/mixed.jpg" width="50%">
  <div class="figcaption">
Another example: Train on rando blends of images, i.e. 40% cat, 60% dog. Then use original images for test.
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/blend.jpg" width="50%">
  <div class="figcaption">
   
     
<a name='simulation'></a>
### 16)	Image simulation
This  paper  ex-ploresdomain  randomization,  a  simple  technique  for  trainingmodels on simulated images that transfer to real images by ran-domizing rendering in the simulator. With enough variability inthe simulator,  the real  world may  appear to  the model  as justanother  variation.  [Josh Tabin, et al, 2017](https://arxiv.org/pdf/1703.06907.pdf)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/simulation.jpg" width="50%">
  <div class="figcaption">

     
<a name='gan'></a>
### 17)	GAN based
This paper uses GAN to generate synthetic medical images. By adding the synthetic dataaugmentation the results significantly increased to 85.7% sensitivity and 92.4% specificity. While the classification performance using only classic data augmentation yielded 78.6%sensitivity and 88.4% specificity. [Maayan Frid-Adar, et al, 2018](https://arxiv.org/pdf/1801.02385.pdf)
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/gan.jpg" width="50%">
  <div class="figcaption">
     
     
<a name='applicable'></a>
## Situations that Data Augmentation is  not Applicable
Not every image data augmentation method can be used for any applications. We need to consider the infomation that the data augmentation added. Sometimes a data augmentation method can mess up the dataset. For example, in numerical OCR model training, we can’t use vertical flipping, because “6” becomes to "9" after vertical flipping.


<a name='package'></a>
## Common image Data Augmentation Packages
-	keras.preprocessing.image.[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
-	[imaug](https://imgaug.readthedocs.io/en/latest/)
-	[albumentations](https://albumentations.ai/)
-	[opencv](https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5)
-	[augmentor](https://augmentor.readthedocs.io/en/master/)
-	[skimage](https://scikit-image.org/docs/dev/api/skimage.html)
-	[solt](https://mipt-oulu.github.io/solt/)
