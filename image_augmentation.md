Table of Contents:

- [Introduction](#intro)
- [Pros and Cons](#pros)
- [Data Augmentation Guidelines](#guidelines)
- [Common Image Augmentation Methods](#methods)
- [Situations not Applicable](#applicable)
- [Common image Augmentation Packages](#package)
- [Examples](#example)
- [Summary](#summary)

<a name='intro'></a>
## Introduction
Data augmentation, the technique of artificially expanding labeled training dataset, is very popular in vision deep learning. It is used as a secret sauce in nearly every state-of-the-art model for image classification, and is becoming increasingly common in other modalities such as natural language understanding as well. More [threories](http://ai.stanford.edu/blog/data-augmentation/) are discussed. 
[Dao et al. 2019](http://proceedings.mlr.press/v97/dao19b/dao19b.pdf) show that a kernel classifier on augmented data approximately decomposes into two components: (i) an averaged version of the transformed features, and (ii) a data-dependent variance regularization term. Data augmentation model combined with a k-nearest neighbor (k-NN) classifier is asymptotically equivalent to a kernel classifier.

<a name='pros'></a>
## Pros and cons
Pros:
-	Induce invariance and regularization, reduce model complexity. Improve generalization. Improve robustness, reduce overfitting. Make a CNN model to be invariant to translation, viewpoint, size or illumination. See [here](https://openreview.net/forum?id=p84tly8c4zf)
-	Artificially increase training and testing dataset even you have a small dataset
-	Improve model generalization

Cons:
-	Too much of image augmentation combined with other forms of regularization (weight L2, dropout) can cause the net to underfit.
-	Too much image augmentation can lead to decreased accuracy in training. 
-	Data augmentation can bring data bias, i.e. the augmented data distribution can be quite different from the original one.See [here]( https://arxiv.org/abs/2010.01267)

<a name='guidelines'></a>
## Data Augmentation Guidelines
-	Data augmentation shall increase in information and a better basis for decision making. 
-	Can focus on the feature and generalize it. See [here](https://arxiv.org/ftp/arxiv/papers/1901/1901.06032.pdf)
-	Avoid data basis which lead to divergence
-	As soon as the transformations can result to an image semantically consistent, i.e. you still can tell it is a cat. 
-	Also can combine these transformations, for example, translate combined with rotate, stretch and shear, 

<a name='methods'></a>
## Common Image Augmentation Methods
1)	Horizontal flips
-	Sometimes vertical flipping is acceptable
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/flip.jpg" width="50%">
  <div class="figcaption">

Krizhevsky et al. 2012, ResNet

2)	Crops/resize/rescale
A section of the image is sampled randomly. 
Training: sample random crops/scales
Resnet:
-	Cat is partially missed in each image
-	Pick random L in range [256,480]
-	Resize training image, short side = L
-	Sample random 224x224 patch
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop1.jpg" width="25%">
  <div class="figcaption">

Testing: average a fixed set of crops
ResNet:
-	Resize image at 5 scales: {224,256,384,480,640}
-	For each size, use 10 224x224 crops: (4 corners + 1 center)  x flips 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop2.jpg" width="50%">
  <div class="figcaption">
 
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/crop3.jpg" width="50%">
  <div class="figcaption">
 
[jchen]

3)	Color jitter
-	Randomize contrast and brightness
-	Apply PCA to all R,G,B channels in training set
-	Sample color offset along principal component directions
-	Add grayscale offset to all pixels of a training image

-	Hue jitter shifts the hue by a random amount
 <div class="fig figcenter fighighlight">
  <img src="/assets/ia/color_jitter.jpg" width="50%">
  <div class="figcaption"> 

4)	Translation
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/translation.jpg" width="50%">
  <div class="figcaption">
   
5)	Rotation 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/rotation.jpg" width="50%">
  <div class="figcaption">
   
6)	Stretching
-	Contrast stretching 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/stretch.jpg" width="50%">
  <div class="figcaption">
   
7)	Shearing
-	To change rectangle image to parallelogram
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/shear.jpg" width="50%">
  <div class="figcaption">
   
8)	Lens distortions
-	In different viewpoint, lens distortion describe the object differently in scale and correlation
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/distortion.jpg" width="50%">
  <div class="figcaption">
[Sebastian Lutz, et al](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1001&context=impstwo)
   
9)	Local warping
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/warp.jpg" width="50%">
  <div class="figcaption">
[Reference1](https://arxiv.org/pdf/1609.08764.pdf)
[Reference2](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14651)
   
10)	Erasing
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/erise.jpg" width="50%">
  <div class="figcaption">
[Reference](https://ojs.aaai.org/index.php/AAAI/article/view/7000)

11)	Contrast / histogram processing 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/contrast.jpg" width="50%">
  <div class="figcaption">

12)	Blur image / add Gaussian noise
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/noise.jpg" width="50%">
  <div class="figcaption">

13)	Using reinforcement learning to do image data augmentation 
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/reinforce.jpg" width="50%">
  <div class="figcaption">
Cubuk et al. AutoAugment: 
Learning Augmentation Strategies from Data, CVPR 2019

14)	 Apply Filter kernel
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/filterKernel.jpg" width="50%">
  <div class="figcaption">
   
15)	 Mix images
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/mixed.jpg" width="50%">
  <div class="figcaption">
   
16)	Image simulation
  <div class="fig figcenter fighighlight">
  <img src="/assets/ia/simulation.jpg" width="50%">
  <div class="figcaption">

<a name='applicable'></a>
## Situations not Applicable
1)	OCR can’t use vertical flipping, because “6” after flipping is “9”.
2)	

<a name='package'></a>
## Common image Augmentation Packages
•	keras.preprocessing.image.ImageDataGenerator
•	imaug
•	albumentations
•	opencv
•	augmentor
•	skimage
•	SOLT
   
<a name='example'></a>
## Examples on how to use image augmentation
-	Procedures on how to choose augmentation methods
-	How to evaluation and test the augmentation
-	Analysis and comments

<a name='summary'></a>
## Summary

   
References:
1)	Krizhevsky et al. 2012, ResNet
2)	Cubuk et al. AutoAugment:  Learning Augmentation Strategies from Data, CVPR 2019
3)	https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0#ref-CR6 
4)	https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/ 
5)	https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/ 
