# Classifying handwritten digits

The objective of this coding exercice is to train a simple neural network on the mnist dataset in order to classify the handwritten digits into numbers ranging from zero to 9.

The problem is a multi-class classification problem on image data.


## Installation

1- Create a virtual env , python version 3.9.16

2- activate virtual env

3- Install dependencies from requirements.txt


```shell
pip install -r requirements.txt

```

4- Run server

In the project : MnistDigistsClassificationWebApp, run main.py

5- Test the model

    a- by running the unit test in testModelAccess.py
    
    b- by uploading a file (sample.png) in the webapp and clicking submit. this page is available at localhost:105 after running main.py
    
6- retraining the model

 by executing the jupyter notebook : training_computervision_mnist.ipynb
 
## Method

- Creating a proof of concept in the notebook training_computervision_mnist.ipynb using google collab.
- Validating the model and testing the prediction in the notebook.
- Saving the model.
- Developping a webapp to server the model results to the end user.

