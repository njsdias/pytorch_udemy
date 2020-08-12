# **Convolution Neural Network**

## Gray Pictures
 These pictures are compounded by pixels. Each pixel has a number between 0 and 255. The back pixel has 0 value and a white pixel has 255 value.

 The **normalization** is a requirement that helps the machine learning algorithm train better. The normalization consists to re-scale the range from 0-255 to 0-1 dived all value by 255.

 The importance of the normalization step is because neural networks rely on gradient calculation helping the consistency of this calculus. These NN are trying to
 calculate how important a certain pixel should be in determining the class image.

 To use the images in a  MLP (Multi Layer Perceptron) we need to convert each image into an array. This processing has a name, **flattening**. This process consist convert a matrix (rows and columns) in an array stacking rows in a vector like: [row1-row2-row3-row4---] in sequential way without - character.

The job of **loss function** is measure any mistakes between a predicted and true class. **Back-propagation** computes the gradient of the loss with respect to the model's weights. In this way we can identify how bad a particular weight is and find which weights in the network are responsible for any errors. The **optimization** step gives us a way to calculate a better weight value. To convert the output layer to a more interpretable layer we use a **softmax** activation function to convert the scores into probabilities.

The loss function used for a multi-class classifier is **categorical cross entropy loss**.

The standard method for minimizing the loss and optimizing for the best weight values is called "Gradient Descent". The are many optimizers:
- SGD
- Momentum
- NAG
- Adagrad
- Adadelta
- Rmsprop

## Cross-Entropy Loss
In the PyTorch documentation, you can see that the cross entropy loss function actually involves two steps:

- It first applies a softmax function to any output is sees
- Then applies NLLLoss; negative log likelihood loss

Then it returns the average loss over a batch of data. Since it applies a softmax function, **we do not have to specify that in the forward function of our model definition**, but we could
define a criterion and use it in the training step.


# **Model Validation**
At advance we don't know how many epochs we need to select to achieve
a good accuracy but not overfitting the training data. The best practice is
split dataset in tthree parts: training, validation and test.

At every training  we check how the model is doing by looking at the training loss and the loss on the validation set. But it's is important to note that he model does not use any part of the validation set for the back propagation step.

We use the training set to find all the patterns we can, and to update and determine the weights of our model.

The validation set only tells us if that model is doing well on the validation set. In this way gives us an idea of how well the model generalizes to a set of data that is separate from the training set.

The idea is use the validation with the weights obtained using training set
to verify is the model is overfitting the training set of data.

The test set is used for checking the accuracy of the model.

# How to check overfitting?

In the graph *loss=f(epoch)* when the curve of validation after decreases loss starts to increases loss with epochs and the training curve stills to decreases
loss with epochs we can concluded the model is not generalize well on validation set.

# Why we need a test set?
The idea is when we go to test the model it looks at data that if has truly never seen before. Even thought the model doesn't use the validation set to update its weights, the model selection process is based on how the model
performs on both the training and validation sets. So, in the end, the model is biased in favor of the validation set. Thus we need a separate test set of data to truly seen how our select model generalizes and performs when given dataset it really has not seen before.

See the code explanation here:

https://www.youtube.com/watch?time_continue=104&v=uGPP_-pbBsc&feature=emb_logo

The code file is : mnist_mlp_solution_with_validation.ipynb)

# Put ideas together

A pipeline that we need follow to build a model can be:
- visualize data
- pre-process: normalize transform
- define a model: do your research
- training your model: define loss & optimization Function
- save the best model: consider using a validation dataset
- test your model


 # **Additional Support**

 Look the follow videos/websites:
 - How computers interprets a gray scale image: https://www.youtube.com/watch?v=mEPfoM68Fx4
 - Understand other codes: https://www.youtube.com/watch?time_continue=28&v=CR4JeAn1fgk&feature=emb_logo
 - Loss&Optimization : https://www.youtube.com/watch?time_continue=25&v=BmPDtSXv18w&feature=emb_logo
 - Training the network : https://www.youtube.com/watch?time_continue=114&v=904bfqibcCw&feature=emb_logo
 - Solution Code Explanation: https://www.youtube.com/watch?time_continue=278&v=7q37WPjQhDA&feature=emb_logo
 - Activation Function website: https://cs231n.github.io/neural-networks-1/#actfun

 # Nomenclature
- **batch size** is the number of the training images that will be seen in one training iteration, where one training iteration means one time that a network make some mistakes and learn from them using back propagation.
- **DataLoader** gives a way to iterate through the data ine batch at a time.
- **epoch number** it is used the training loop. It defines how many times we want model to iterate to entire training dataset.
