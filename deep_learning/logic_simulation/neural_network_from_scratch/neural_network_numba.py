#!/usr/bin/env python3
import math
import random
import numpy as np
from numba import cuda, float32

# -----------------------------------------------------------------------------
# CUDA kernel implementing the entire training loop with in‑kernel debug printing.
# Assumptions:
#   - The training set is small enough to fit into one block.
#   - Each thread handles one training sample per epoch.
#   - Weight updates are performed using atomic adds.
@cuda.jit
def train_full_kernel(training_data, labels, wih, who,
                      num_samples, input_nodes, hidden_nodes,
                      alpha, epochs):
    tid = cuda.threadIdx.x  # Each thread corresponds to one training sample.
    # Local storage for hidden layer activations (assumes hidden_nodes <= 128)
    hidden = cuda.local.array(128, dtype=float32)
    
    for ep in range(epochs):
        if tid < num_samples:
            # ---- Forward Propagation for sample 'tid' ----
            for i in range(hidden_nodes):
                s = 0.0
                for j in range(input_nodes):
                    s += training_data[tid, j] * wih[i, j]
                hidden[i] = 1.0 / (1.0 + math.exp(-s))
            out_sum = 0.0
            for i in range(hidden_nodes):
                out_sum += hidden[i] * who[i]
            output = 1.0 / (1.0 + math.exp(-out_sum))
            
            # ---- Backpropagation ----
            error = labels[tid] - output
            d_out = output * (1.0 - output)
            delta = error * d_out
            # Update hidden-to-output weights atomically.
            for i in range(hidden_nodes):
                cuda.atomic.add(who, i, alpha * delta * hidden[i])
            # Update input-to-hidden weights atomically.
            for i in range(hidden_nodes):
                d_hidden = hidden[i] * (1.0 - hidden[i])
                for j in range(input_nodes):
                    cuda.atomic.add(wih, (i, j),
                                    alpha * delta * who[i] * d_hidden * training_data[tid, j])
        # Synchronize all threads within the block at the end of each epoch.
        cuda.syncthreads()
        # Debug output: only thread 0 prints cost every 1000 epochs.
        if tid == 0 and (ep % 100 == 0):
            # Recompute forward pass for the first sample (sample 0)
            for i in range(hidden_nodes):
                tmp = 0.0
                for j in range(input_nodes):
                    tmp += training_data[0, j] * wih[i, j]
                hidden[i] = 1.0 / (1.0 + math.exp(-tmp))
            out_sum = 0.0
            for i in range(hidden_nodes):
                out_sum += hidden[i] * who[i]
            output = 1.0 / (1.0 + math.exp(-out_sum))
            cost = (labels[0] - output) ** 2
            print("Epoch =", ep, "Cost =", cost)

# -----------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, alpha=0.01, epochs=1000, use_gpu=True):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.alpha = alpha
        self.epochs = epochs
        self.use_gpu = use_gpu

        # Initialize weight matrices in float32.
        self.wih = np.random.uniform(-1, 1, (hidden_nodes, input_nodes)).astype(np.float32)
        self.who = np.random.uniform(-1, 1, hidden_nodes).astype(np.float32)

    def safe_sigmoid(self, s):
        # Clip s to avoid overflow in exp.
        if s < -50:
            return 0.0
        elif s > 50:
            return 1.0
        else:
            return 1.0 / (1.0 + math.exp(-s))

    def forward_prop(self, inputs):
        """CPU forward propagation (for prediction)."""
        x = np.array(inputs, dtype=np.float32)
        hidden = [0.0] * self.hidden_nodes
        for i in range(self.hidden_nodes):
            s = 0.0
            for j in range(self.input_nodes):
                s += x[j] * self.wih[i, j]
            hidden[i] = self.safe_sigmoid(s)
        out_sum = 0.0
        for i in range(self.hidden_nodes):
            out_sum += hidden[i] * self.who[i]
        output = self.safe_sigmoid(out_sum)
        return output

    def train_cpu(self, training_data, labels):
        """Simple CPU training loop (stochastic gradient descent)."""
        num_samples = len(training_data)
        for ep in range(self.epochs):
            for i in range(num_samples):
                x = np.array(training_data[i], dtype=np.float32)
                target = labels[i]
                hidden = [0.0] * self.hidden_nodes
                for h in range(self.hidden_nodes):
                    s = 0.0
                    for j in range(self.input_nodes):
                        s += x[j] * self.wih[h, j]
                    hidden[h] = 1.0 / (1.0 + math.exp(-s))
                out_sum = 0.0
                for h in range(self.hidden_nodes):
                    out_sum += hidden[h] * self.who[h]
                output = 1.0 / (1.0 + math.exp(-out_sum))
                error = target - output
                d_out = output * (1.0 - output)
                delta = error * d_out
                for h in range(self.hidden_nodes):
                    self.who[h] += self.alpha * delta * hidden[h]
                for h in range(self.hidden_nodes):
                    d_hidden = hidden[h] * (1.0 - hidden[h])
                    for j in range(self.input_nodes):
                        self.wih[h, j] += self.alpha * delta * self.who[h] * d_hidden * x[j]
            if ep % 1000 == 0:
                out_val = self.forward_prop(training_data[0])
                cost_val = (labels[0] - out_val) ** 2
                print(f"Epoch = {ep}  Cost = {cost_val:.4f}")

    def train_gpu_full(self, training_data, labels):
        """
        Train the network entirely on the GPU using the Numba CUDA kernel.
        Assumes the number of training samples is small enough to fit in one block.
        """
        num_samples = len(training_data)
        train_np = np.array(training_data, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        
        # Copy data and weights to the GPU.
        d_train = cuda.to_device(train_np)
        d_labels = cuda.to_device(labels_np)
        d_wih = cuda.to_device(self.wih)
        d_who = cuda.to_device(self.who)
        
        # Set grid and block sizes.
        block_size = 1024  # threads per block
        grid_size = 15   # blocks per grid (adjustable for your 4070)
        
        train_full_kernel[grid_size, block_size](d_train, d_labels, d_wih, d_who, 
                                                  np.int32(num_samples),
                                                  np.int32(self.input_nodes),
                                                  np.int32(self.hidden_nodes),
                                                  np.float32(self.alpha),
                                                  np.int32(self.epochs))
        cuda.synchronize()
        # Copy updated weights back to host.
        self.wih = d_wih.copy_to_host()
        self.who = d_who.copy_to_host()

    def train(self, training_data, labels):
        """Dispatch training based on use_gpu flag."""
        if self.use_gpu:
            self.train_gpu_full(training_data, labels)
        else:
            self.train_cpu(training_data, labels)

    def predict(self, inputs):
        """Return binary prediction (0 or 1) based on threshold 0.5."""
        output = self.forward_prop(inputs)
        return 1 if output > 0.5 else 0

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # Generate 100 training examples: label = 1 if (A - (B+C) > 0), else 0.
    training_data = []
    training_labels = []
    for _ in range(1000):
        a_val = random.randint(0, 20)
        b_val = random.randint(0, 10)
        c_val = random.randint(0, 10)
        training_data.append([a_val, b_val, c_val])
        training_labels.append(1 if (a_val - (b_val + c_val)) > 0 else 0)
    
    # Shuffle and split dataset (80/20).
    combined = list(zip(training_data, training_labels))
    random.shuffle(combined)
    training_data, training_labels = zip(*combined)
    training_data = list(training_data)
    training_labels = list(training_labels)
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    train_labels = training_labels[:split_idx]
    test_data = training_data[split_idx:]
    test_labels = training_labels[split_idx:]
    
    print(f"Training on {len(train_data)} samples; Testing on {len(test_data)} samples.")
    print("First 5 training examples:")
    for i in range(5):
        print(f"Sample: {train_data[i]}  Label: {train_labels[i]}")
    
    # Create the network.
    # Set use_gpu to True to run training on the GPU using Numba or False for CPU training.
    model = NeuralNetwork(input_nodes=3, hidden_nodes=6, alpha=0.001, epochs=1000, use_gpu=True)
    model.train(train_data, train_labels)
    
    # Evaluate on the test set.
    correct = 0
    for sample, label in zip(test_data, test_labels):
        if model.predict(sample) == label:
            correct += 1
    accuracy = correct / len(test_data)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
