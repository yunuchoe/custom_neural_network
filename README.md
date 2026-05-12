# Neural Network

This repository contains an object oriented framework designed to solve Gaussian XOR classification problems.

This was completed as an academic assignment for ECE 471 (Computer Vision) to demonstrate a mathematical understanding of computational graphs and backpropagation. Here is the distribution of code: 
* **Provided:** The class structures, intermediate test cases, and the Gaussian XOR data generation.
* **Implemented:** The forward and backward passes, gradient accumulation, parameter stepping and final testing.

### Network Architecture
The framework builds a computational graph consisting of:
* **Mathematical Nodes:** Multiplication, Addition, and AddConstant
* **Loss & Activation:** Squared Error loss and Sigmoid activation functions
* **Design:** A 2-Input -> 2-Hidden -> 1-Output

### Training Dynamics & Stochastic Variance
Because the network relies on random Gaussian distributions for both parameter initialization and dataset generation without a fixed seed, the model demonstrates the natural variance inherent to stochastic gradient descent. 

Running the training execution loop across optimized hyperparameters (Learning Rate: 1.5, Epochs: 200) yields varying decision boundary formations. As a result, final test accuracies fluctuate per run, with 93% being a rough average, reflecting the stochastic nature of the unseeded initialization.
