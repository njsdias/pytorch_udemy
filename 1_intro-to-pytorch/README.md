# **Nomenclature**
**epoch:** some pass through the entire dataset is called an epoch

**Validation:** measures our model's performance on data that is not part of our training set. In other words, the performance is measured using validation dataset.

**batch_size:** number of images/examples you get per batch like per loop
through

**shuffle:** randomly shuffles your data every time you start a new epoch. It is
useful because when you're training your network because every time the images are in the different order.

**data augumentation:** it helps network generalize because with augumentation
we can randomly rotate, mirror, scale, and crop during the training. For testing/validation we don't
do data augumentation. We only need to resize and center crop your images.
This is because you want the validation is more like and input picture (without manipulation).

**Measure the model's performance**: recall, precision, top k error rate

# **Backward propagation**

## **Training in Pytorch**

**Part 3**

It is used to calculate gradients when we calculate the loss. Our loss, depends on our weight and bias parameters. We need gradients to the gradients of our weights to do gradient descent.

We can do is, set up our weights as tensors taht requires gradients and then do a froward pass to calculate our loss. With the loss, you do a backwards pass which calculates the gradients for our weights. With those gradients, you can do your gradient descent step.

To use these gradients to calculate gradient descent we use optimize package from Pytroch: `optim.SGD(model.parameters(), lr = 0.01)`

**Training process for each training batch**

1- get data which corresponds at each batch: `for images, labels in trainloader`
2- clean the gradients
3- Make a forward pass through the network
4- Use the network output to calculate the loss
5- Perform a backward pass through the network with loss.backward() to calculate the gradients
6- Take a step with the optimizer to update the weights

Do not forget to clear the gradients because the PyTorch accumulates them. So, at each step if you don't clear the gradients PyTorch sum it with previous training step and at the end you don't training properly the neural network.

```
for images, labels in trainloader:
    # zeroing grad
    optimizer.zero_grad()

    # forward model
    log_ps = model(images)
    # calculate loss
    loss = criterion(log_ps, labels)
    # getting the gradients
    loss.backward()
    # using gradient to calculate SGD
    optimizer.step()
```

# **Inference and Validation**
**Part 5**

To overcome the overfitting problem is usual use regularization such as dropout.

`self.dropout = nn.Dropout(p=0.2)`

as `p=0.2` the probability that you'll drop a unit.

In the last layer we don't use dropout.



In validation we don't need to do any training and we will turn off dropout:

```
with torch.no_grad():
  # set model to evaluation model
  mode.eval()

  # validation pass here
  for images, labels in testloader:
    ...

# set mode back to train model
mode.train()
```

In general, **the pattern for the validation loop** will look like this
- turn off gradients
- set the model to evaluation mode
- calculate the validation loss and metric
- then set the model back to train mode.


# **Saving and Loading model**
**Part 6**

To save model we use:
- `torch.save(model.state_dict(), 'checkpoint.pth')`

To load model we use:
```
state_dict = torch.load(checkpoint.pth)`
model.load_state_dict(state_dict)
```

The above method only results if the new model that we use has the same
architecture than the model that was saved.

To overcome this we need to include the information of the model in the
checkpoint. For instance, to save and load:

```
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model
```

# **Loading Image dataset**

If you fave a folder with images to train and test and you want to use `ImageFolder`
you need to put your image files inside the `test` and `train` folders like this:

`datsets/train_data/train`
`datasets/test_data/test`

and you use the below lines
```
data_dir = 'datasets/'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

# We only need to resize and center crop our test/validation images
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(data_dir + '/train_data/', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test_data/', transform=test_transforms)
```

## **Using Pre-Trained Model**
**Part 8**

A pre-trained model was trained using an specific architecture. In general,
this model needs to be retrained at classifier level. For instance,
the Densenet121, which has 1212 layers, has a classifier with 100 different
categories. If our dataset only has 2 different categories we need retrain
the classifier for 2 categories maintain the features part static.
```
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)), # we only have two labels
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
```

This new classifier is untrained and is attached to our model and this model also has the features parts. The features parts are going to remain frozen. We're not
update those weights but we need to train our new classifier. Use a GPU for that.

# Watch those shapes
In general, you'll want to check that the tensors going through your model and other code are the correct shapes. Make use of the `.shape` method during debugging and development.

## A few things to check if your network isn't training appropriately
Make sure you're clearing the gradients in the training loop with `optimizer.zero_grad()`. If you're doing a validation loop, be sure to set the network to evaluation mode with `model.eval()`, then back to training mode with `model.train()`.

## CUDA errors
Sometimes you'll see this error:

`RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #1 ‘mat1’`

You'll notice the second type is `torch.cuda.FloatTensor`, this means it's a tensor that has been moved to the GPU. It's expecting a tensor with type torch.FloatTensor, no `.cuda` there, which means the tensor should be on the CPU. PyTorch can only perform operations on tensors that are on the same device, so either both CPU or both GPU. If you're trying to run your network on the GPU, check to make sure you've moved the model and all necessary tensors to the GPU with `.to(device)` where device is either "cuda" or "cpu".
