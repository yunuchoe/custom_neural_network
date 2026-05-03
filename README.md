# Custom Deep Learning Framework: Autograd & MLP

This repository contains an object-oriented deep learning framework built with just standard Python, designed to solve the non-linear Gaussian XOR classification problem.

This was completed as an academic assignment for ECE 471 (Computer Vision) to demonstrate a mathematical understanding of computational graphs and backpropagation without the use of libraries like PyTorch or TensorFlow. Here is the distribution of code: 
* **Provided:** The fundamental class structures (`Node` base class), testing loop scaffolding, and the synthetic Gaussian XOR data generation functions were provided by the university.
* **Implemented:** The core mathematical engines—specifically the `forward` pass computations, the `backward` pass derivative calculus (chain rule routing), gradient accumulation, and parameter stepping—were entirely manually derived and engineered.

### Network Architecture
The framework builds a dynamically updating computational graph consisting of:
* **Mathematical Nodes:** Multiplication, Addition, and AddConstant.
* **Loss & Activation:** Squared Error loss and Sigmoid activation functions.
* **Topology:** A 2-Input $\rightarrow$ 2-Hidden $\rightarrow$ 1-Output Multilayer Perceptron (MLP).

### Training Dynamics & Stochastic Variance
Because the network relies on random Gaussian distributions for both parameter initialization and synthetic dataset generation without a fixed random seed, the model demonstrates the natural variance inherent to stochastic gradient descent. 

Running the training execution loop across optimized hyperparameters (e.g., Learning Rate: 1.5, Epochs: 200) yields varying decision boundary formations. As a result, final test accuracies naturally fluctuate per run, with 93% being a rough average, accurately reflecting the stochasticity of the unseeded initialization.
