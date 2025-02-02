## Linear Classifiers for image classification

Overview. I have developed a more powerful approach to image classification than KNN that will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks. The approach has two major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. It then casts this as an optimization problem in which minimizes the loss function with respect to the parameters of the score function. (Pretty much same to any deep learning algorithms). Here i also introducet the idea of regularization (L1, L2, Elastic net (L1+L2)), In next projects I added (Dropout and Batch Norm)

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/svmvssoftmax.png" width=600 height=400>

## Support Vector Machine

The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. (Uses hinge loss)

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/SVMp.png" width=400 height=400>

## Softmax

the Softmax classifier which has a different loss function than SVM. If you’ve heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes. In the Softmax classifier, the function mapping f(xi;W)=Wxi
stays unchanged, but now it is interpreted these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss.

<img src="https://github.com/shushukurov/ML_Portfolio/blob/main/LinearClassifiers_SVM_Softmax/Softmax.jpg" width=600 height=400>
