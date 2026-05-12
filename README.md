# Neural Network

This repository contains an object-oriented framework designed to solve Gaussian XOR classification problems.

This was completed as an academic assignment for ECE 471 (Computer Vision) to demonstrate a mathematical understanding of computational graphs and backpropagation without the use of libraries like PyTorch or TensorFlow. Here is the distribution of code: 
* **Provided:** The fundamental class structures (Node base class), testing loop, and the Gaussian XOR data generation functions.
* **Implemented:** The core mathematical engines, specifically the forward and backward passes, gradient accumulation, and parameter stepping.

### Network Architecture
The framework builds a computational graph consisting of:
* **Mathematical Nodes:** Multiplication, Addition, and AddConstant.
* **Loss & Activation:** Squared Error loss and Sigmoid activation functions.
* **Design:** A 2-Input -> 2-Hidden -> 1-Output

### Training Dynamics & Stochastic Variance
Because the network relies on random Gaussian distributions for both parameter initialization and dataset generation without a fixed seed, the model demonstrates the natural variance inherent to stochastic gradient descent. 

Running the training execution loop across optimized hyperparameters (Learning Rate: 1.5, Epochs: 200) yields varying decision boundary formations. As a result, final test accuracies naturally fluctuate per run, with 93% being a rough average, reflecting the stochastic nature of the unseeded initialization.
