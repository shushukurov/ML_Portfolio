# Portfolio
List of My projects
Inspired by Andrej Karpathy, Justin Johnson, Cs231n, Elon Musk's tweet about PyTorch :)
Most projects are implented on low level using Pytorch tensors only as a gpu accelerating data type, only few of them (most complicated ones) utilize some pytorch high level API functions 

## 1. [KNN for Image Classification](https://github.com/shushukurov/ML_Portfolio/tree/main/KNN_for_ImageClassification)
KNN classifier for CIFAR-10 from scratch with PyTorch. KNN is data driven,
image classification algorithm that was popular before Deep Learning came out. 
So I structured my PyTorch porfolio projects according to timeline of algorithms were developed (Popular)
So I have started with KNN

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/KNN_for_ImageClassification/KNN.png" width=400 height=400>

## 2. [Linear Classifiers for image classification](https://github.com/shushukurov/ML_Portfolio/tree/main/LinearClassifiers_SVM_Softmax)

Overview. I have developed a more powerful approach to image classification than KNN that will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks. The approach has two major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. It then casts this as an optimization problem in which minimizes the loss function with respect to the parameters of the score function. (Pretty much same to any deep learning algorithms). Here i also introducet the idea of regularization (L1, L2, Elastic net (L1+L2)), In next projects I added (Dropout and Batch Norm)

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/svmvssoftmax.png" width=600 height=400>

## Support Vector Machine

The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. (Uses hinge loss)

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/SVMp.png" width=400 height=400>

## Softmax

the Softmax classifier which has a different loss function than SVM. If you’ve heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes. In the Softmax classifier, the function mapping f(xi;W)=Wxi
stays unchanged, but now it is interpreted these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/Softmax.jpg" width=600 height=400>

## 3. [Deep Learning Modules + Functions implementation](https://github.com/shushukurov/ML_Portfolio/tree/main/DeepLearningModule)

This Project consists of Modular implementation of Fully-Connected Neural Networks, Dropout and different optimizers (SGD, Momentum, Rmsprop, Adam)

## Modular implementation of Fully-Connected Neural Networks
Andrej Karphathy once wrote that ML engineers should have deep understanding of backpropagation. Therefore I implemented all necessary modules for Neural Networks from scratch using PyTorch GPU acceleration in order Improve knowledge of NN and Backprop.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/DeepLearningModule/Backprop.png" width=800 height=400>

## Dropout

Dropout is a technique for regularizing neural networks by randomly setting some output activations to zero during the forward pass.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/DeepLearningModule/Dropout.png" width=600 height=400>

## Optimizers (SGD, Momentum, Rmsprop, Adam)

So far I have used vanilla stochastic gradient descent (SGD) as my update rule. More sophisticated update rules can make it easier to train deep networks. Therefore I have implement a few of the most commonly used update rules (SGD, Momentum, RMsprop, Adam) and compare them to vanilla SGD.

![Alt Text](https://github.com/shushukurov/ML_Portfolio/blob/main/DeepLearningModule/Optimizers.png)

## 4. [Convolutional Neural Network, Batch Normalization and Kaiming initialization](https://github.com/shushukurov/ML_Portfolio/tree/main/CNN_BatchNorm_Kaiming)

In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/CNN_BatchNorm_Kaiming/CIFAR10_CNN.gif" width=700 height=400>

## Batch Norm
Batch normalization (also known as batch norm) is a method used to make artificial neural networks faster and more stable through normalization of the input layer by re-centering and re-scaling. It was proposed by Sergey Ioffe and Christian Szegedy in 2015.

<img src="https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/batchnorm_graph.png" width=691 height=202>


## Kaiming Initialization
Kaiming Initialization, or He Initialization, is an initialization method for neural networks that takes into account the non-linearity of activation functions, such as ReLU activations.



## 5. [Image Captioning (RNN, LSTM, Attention)](https://github.com/shushukurov/ML_Portfolio/tree/main/ImageCaptioning)

Generally, a captioning model is a combination of two separate architecture that is CNN (Convolutional Neural Networks)& RNN (Recurrent Neural Networks) and in this case LSTM (Long Short Term Memory), which is a special kind of RNN that includes a memory cell, in order to maintain the information for a longer period. 

## Recurrent Neural Network
Recurrent Neural Network (RNN) language models for image captioning.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/RNN.png" width=404 height=404>

## Long-Short-Term-Memory
Many people use a variant on the vanilla RNN called Long-Short Term Memory (LSTM) RNNs because Vanilla RNNs can be tough to train on long sequences due to vanishing and exploding gradients caused by repeated matrix multiplication. LSTMs solve this problem by replacing the simple update rule of the vanilla RNN with a gating mechanism as follows.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/LSTM.png" width=404 height=404>

## Attention
Attention is the idea of freeing the encoder-decoder architecture from the fixed-length internal representation.
This is achieved by keeping the intermediate outputs from the encoder LSTM from each step of the input sequence and training the model to learn to pay selective attention to these inputs and relate them to items in the output sequence.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/ImageCaptioning/Attention.png" width=691 height=404>

## 6. [Neural Networks Visualization](https://github.com/shushukurov/ML_Portfolio/tree/main/NetworkVisualization)

## Saliency Maps

A saliency map tells the degree to which each pixel in the image affects the classification score for that image.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/NetworkVisualization/SilencyMaps.jpg" width=691 height=404>

## Adversarial Attacks

Use of image gradients to generate "adversarial attacks". Given an image and a target class, It is possible to perform gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/NetworkVisualization/AdversarialAttack.jpeg" width=500 height=404>

## Class visualization

By starting with a random noise image and performing gradient ascent on a target class, It is possible to generate an image that the network will recognize as the target class.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/NetworkVisualization/Class_Visual.png" width=404 height=404>

## 7. Style Transfer (Soon)

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation.

![caption](http://web.eecs.umich.edu/~justincj/teaching/eecs498/example_styletransfer.png)


## 8. [Single-Stage Object Detector (YOLO 2)](https://github.com/shushukurov/ML_Portfolio/tree/main/SingleStageDetector_YOLO)

In project I implemented a single-stage object detector, 
based on YOLO (v1 and v2) and used it to train a model that can detect objects on novel images. 
I also evaluated the detection accuracy using the classic metric mean Average Precision (mAP). 
In Next project (That extends this project), I will implement a two-stage object detector, based on Faster R-CNN. 
The main difference between the two is that single-stage detectors perform region proposal and 
classification simultaneously while two-stage detectors have them decoupled.

![alt text](https://github.com/shushukurov/ML_Portfolio/blob/main/SingleStageDetector_YOLO/OPGDq.jpg)

## 9. [Double-Stage Object Detector (Faster R-CNN)](https://github.com/shushukurov/ML_Portfolio/tree/main/FasterRCNN)

In this project I implemented a two-stage object detector, based on Faster R-CNN, which consists of two modules, Region Proposal Networks (RPN) and Fast R-CNN and extends previous project YOLO detector. I use it to train a model that can detect objects on novel images and evaluate the detection accuracy using the classic metric mean Average Precision (mAP)

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/FasterRCNN/FasterRCNN.png" width=600 height=400>

## 10. Generative Adversarial Network
(Soon)


## 11. [Resnet (Pre-Resnet + BottleNeck block)](https://github.com/shushukurov/ML_Portfolio/tree/main/Resnet)

This project is motivated by my desire to deeply understand the state of art architecutre 'Residual network' AKA 'Resnet'. When I was studying CNNs for visual recognition I always tried to understand papers first then re-implement by myself using favourite tools e.g Pytorch or Numpy. However, I could not fint any tutorial to re-implement Resnet based architures from scratch using pytorch (All I could find using PyTorch Implemented models from Model Zoo) So I decided to try myself mainly by looking on CS231n lecture notes (Mainly raw NumPy based code) and re-implent it using only PyTorch

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/Resnet/resnet.png" width=400 height=600>
