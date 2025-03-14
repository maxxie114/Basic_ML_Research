# Neural Network from Scratch with GPU Support

## Overview

This project implements a simple feedforward neural network entirely from scratch in Python. The network is designed for a binary classification task and includes GPU support via PyCUDA. Its purpose is to illustrate the basic principles behind neural networks—including the math behind forward propagation, back propagation, and weight updates—while demonstrating how to accelerate computations using a GPU.

The network consists of:
- **Input Layer:** Accepts a vector of features (e.g., A, B, C). The number of input nodes is configurable.
- **Hidden Layer:** A configurable number of neurons (default is 3). Each hidden neuron receives inputs from every feature.
- **Output Layer:** A single neuron that outputs a probability (via the sigmoid activation), which we threshold to produce a classification.

The project also includes:
- Dynamic initialization of weights stored in Python lists.
- An 80/20 train-test split of a generated dataset (100 samples) based on the rule: label = 1 if (A – (B + C) > 0), otherwise 0.
- Evaluation of model accuracy on the test set.
- GPU support that leverages PyCUDA for performing forward and back propagation in parallel.

## Network Architecture

### Layers

1. **Input Layer:**  
   Receives the feature vector. For example, if using three features, the input is:  
   \[ A, B, C \]

2. **Hidden Layer:**  
   Consists of *m* neurons (e.g., 3 neurons by default). Each hidden neuron is connected to every input.  
   The weights between the input layer and the hidden layer are stored as a 2D matrix (each row represents a hidden neuron’s weights):  
   ```
   weights_input_hidden = [
       [w₁₁, w₁₂, w₁₃],  // Weights for Hidden Neuron 1
       [w₂₁, w₂₂, w₂₃],  // Weights for Hidden Neuron 2
       [w₃₁, w₃₂, w₃₃]   // Weights for Hidden Neuron 3
   ]
   ```
   For each hidden neuron *i*, the weighted sum is computed as:  
    zᵢ = wᵢ₁·x₁ + wᵢ₂·x₂ + wᵢ₃·x₃  
   and the activation is:  
    aᵢ = sigmoid(zᵢ)

3. **Output Layer:**  
   A single neuron aggregates the outputs from the hidden layer. The connections between the hidden layer and the output layer are stored as a 1D list (one weight per hidden neuron):  
   ```
   weights_hidden_output = [v₁, v₂, v₃]
   ```
   The output neuron computes:  
    z_out = v₁·a₁ + v₂·a₂ + v₃·a₃  
   Then, the final prediction is:  
    ŷ = sigmoid(z_out)

### Diagram

```
                     [ Input Layer ]
                         x₁   x₂   x₃
                           \  |  /
                            \ | /
            +--------------------------------+
            |      Hidden Layer (3 neurons)  |
            |   Neuron 1: a₁ = sigmoid(z₁)     |  <- weights: [w₁₁, w₁₂, w₁₃]
            |   Neuron 2: a₂ = sigmoid(z₂)     |  <- weights: [w₂₁, w₂₂, w₂₃]
            |   Neuron 3: a₃ = sigmoid(z₃)     |  <- weights: [w₃₁, w₃₂, w₃₃]
            +--------------------------------+
                         |    |    |
                      (v₁)|(v₂)|(v₃)
                         \    |    /
                          \   |   /
                     +----------------+
                     |  Output Layer  |
                     |  ŷ = sigmoid   |
                     +----------------+
```

## Mathematical Details

### Sigmoid Activation

The **sigmoid function** is given by:
\[
\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
\]
Its derivative, needed for back propagation, is:
\[
\text{sigmoid}'(z) = \text{sigmoid}(z) \times (1 - \text{sigmoid}(z))
\]

### Forward Propagation

- **Hidden Layer:**  
  For each hidden neuron *i*, calculate:
  \[
  z_i = \sum_{j=1}^{n} \left(w_{ij} \times x_j \right)
  \]
  \[
  a_i = \text{sigmoid}(z_i)
  \]

- **Output Layer:**  
  Calculate the output neuron’s weighted sum:
  \[
  z_{\text{out}} = \sum_{i=1}^{m} \left(v_i \times a_i\right)
  \]
  Then compute the prediction:
  \[
  \hat{y} = \text{sigmoid}(z_{\text{out}})
  \]

### Cost Function

We use the **squared error** cost:
\[
\text{Cost} = (y_{\text{actual}} - \hat{y})^2
\]

### Back Propagation

Weights are updated using gradient descent.

- **For the hidden-to-output weights (vᵢ):**
  \[
  \Delta v_i = \alpha \times 2 \times (y_{\text{actual}} - \hat{y}) \times \hat{y}(1 - \hat{y}) \times a_i
  \]

- **For the input-to-hidden weights (wᵢⱼ):**
  \[
  \Delta w_{ij} = \alpha \times 2 \times (y_{\text{actual}} - \hat{y}) \times \hat{y}(1 - \hat{y}) \times v_i \times a_i(1 - a_i) \times x_j
  \]

Here, \(\alpha\) is the learning rate.

## GPU Acceleration with PyCUDA

When using GPU support:
- The training data and weight matrices are converted to flat NumPy arrays and transferred to GPU memory.
- A CUDA kernel (written in CUDA C) performs forward and back propagation on each training sample in parallel.
- Atomic operations update the weights safely from multiple threads.
- A delay (e.g., 0.5 seconds) is inserted after each epoch to allow you to monitor GPU usage.

## Training and Evaluation Process

1. **Dataset Generation:**  
   100 samples are generated using the rule:
   - Let A be randomly chosen from 0 to 20.
   - Let B and C be randomly chosen from 0 to 10.
   - Label = 1 if \( A - (B + C) > 0 \); otherwise, label = 0.

2. **Train-Test Split:**  
   The 100 samples are shuffled and split into:
   - 80 training samples.
   - 20 testing samples.

3. **Model Training and Evaluation:**  
   - The network is trained on the training set (using either the CPU or GPU branch).
   - After training, predictions are made on the test set.
   - Accuracy is computed by comparing the predicted labels to the actual labels.

## How to Use

1. **Installation:**  
   Ensure you have Python 3.x, NumPy, and PyCUDA installed.  
   (PyCUDA is required for GPU support.)

2. **Running the Code:**  
   - Clone the repository.
   - Run the main Python script.  
   The script will train the network, then perform an evaluation on the test set and display the accuracy.

3. **Modifying the Network:**  
   - You can change the number of input nodes and hidden neurons by modifying the parameters in the `NeuralNetwork` constructor.
   - You can switch GPU mode on or off via the `use_gpu` parameter.

## Conclusion

This project is a beginner-friendly introduction to building a simple neural network from scratch. It covers:
- The structure and flow of a feedforward neural network.
- The mathematical foundations behind the forward and backward passes.
- How to update weights using gradient descent.
- How to leverage GPU acceleration with PyCUDA.
- How to evaluate model performance with a train-test split.

Feel free to explore, modify, and experiment with the code to deepen your understanding of neural networks!

─────────────────────────────  
Enjoy learning and happy coding!
─────────────────────────────  