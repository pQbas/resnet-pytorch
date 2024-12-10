<div align="center">
  <h1>resnet-pytorch</h1>
  ResNet CNN implemented in PyTorch, ready for training, testing, and inference.
  <!--  -->
  <!-- <p align="center"> -->
  <!--   <a href="here_is_a_demo_video"> -->
  <!--   <img alt="Blueberry Detection ROS" src="gallery/image-demo.png"></a> -->
  </p>

</div>


## How install?

Go to the folder of this project and run pip install.
```
cd resnet-pytorch
pip install -e .
```

## How use it?

This library gives access for three main actions with the resnet-cnn, this actions are
`train`, `test` and `inference`. The `demo` folder contains an example of how use it
with a notebook ready to use in colab. Below are some snippets wich explains the code 
in the demo folder.


### Train action

Following code helps you to train resnet. To train is needed to define a `CONFIG_PARAMS`
constant, this is a dictionary that contains training parameters such as `batch size`,
`categories`, `optimizer`, `learning rate`, etc. The `train` function receives this
dictionary and gives you the path where the weights were saved as a `pt` file.

```python
# Import resnet library previously installed
import resnet

# Define the config params for all proceess
CONFIG_PARAMS = {
    'batch_size'    : 16,
    'categories'    : 10,
    'optimizer'     : 'sgd',
    'learning_rate' : 0.001,
    'loss'          : 'cross-entropy',
    'epochs'        : 5,
    'model_name'    : 'resnet',
    'path'          : 'runs',
    'dataset_name'  : 'cifar10',
}

# Train the resnet model
weightsPath = resnet.train(params = CONFIG_PARAMS)
```

### Test action

Result of this action is the accuracy metric computed for the trained model, this
function receives the `params` paramtere and also the `weights path`.

```python
# Import resnet library previously installed
import resnet

# Test the ResNet model
accuracy = resnet.test(params      = CONFIG_PARAMS, 
                       weightsPath = weightsPath)
```

### Inference action

Inference receives an image, model and the device as input, and gives you the category
of the image. In following example is used PIL to load the image, and some utilities
as for loading the model and getting the device. 

```python
# Import resnet library previously installed
import resnet
from PIL import Image

# Constat with an image to perform the testing
IMG_PATH = '../gallery/cat.jpeg'

# Getting the main device to perform inference `gpu` by defult.
DEVICE = resnet.utils.getDevice()

# Load model the trained model and image 
model = resnet.utils.loadModel(weightsPath = weightsPath, 
                               params      = CONFIG_PARAMS, 
                               device      = DEVICE)
image = Image.open(IMG_PATH)

# Perform inference (preprocessing and prediction)
results = resnet.inference(image, model, DEVICE)
```
