#!/usr/bin/env python3
import math
import random

class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, alpha=0.01, epoch=1000, use_gpu=False):
        # Allow a customizable number of input features (formerly a, b, c) and hidden neurons.
        self.input_nodes = input_nodes      # For example, originally a, b, c
        self.hidden_nodes = hidden_nodes    # Each hidden neuron corresponds to weights like [aw, bw, cw]
        self.alpha = alpha
        self.epoch = epoch
        self.use_gpu = use_gpu
        # Dynamically create the input-to-hidden weight matrix (replacing aw1, bw1, cw1, etc.).
        self.weights_input_hidden = [
            [random.uniform(-1, 1) for _ in range(self.input_nodes)]
            for _ in range(self.hidden_nodes)
        ]
        # Create the hidden-to-output weights (replacing xw, yw, zw).
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(self.hidden_nodes)]
        # Placeholders for intermediate values during forward propagation.
        self.hidden_activations = [0.0] * self.hidden_nodes
        self.last_input = [0.0] * self.input_nodes
        self.output = 0.0

    def __sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def forward_prop(self, inputs):
        """
        Performs forward propagation.
        For each hidden neuron, computes the weighted sum (like a*aw + b*bw + c*cw) and applies sigmoid.
        Then computes the output using the hidden-to-output weights (like x*xw + y*yw + z*zw).
        """
        hidden = []
        for i in range(self.hidden_nodes):
            z = 0.0
            for j in range(self.input_nodes):
                z += inputs[j] * self.weights_input_hidden[i][j]
            hidden.append(self.__sigmoid(z))
        out_sum = 0.0
        for i in range(self.hidden_nodes):
            out_sum += hidden[i] * self.weights_hidden_output[i]
        output = self.__sigmoid(out_sum)
        # Save intermediate values (similar to saving a, b, c and x, y, z in your original code).
        self.hidden_activations = hidden
        self.last_input = inputs
        self.output = output
        return output

    def __d_sigmoid(self, sx):
        return sx * (1 - sx)

    def __d_cost(self, actual):
        return 2 * (actual - self.output)

    def back_prop(self, actual):
        """
        Performs back propagation. Updates:
         • hidden-to-output weights (like xw, yw, zw) using the output error and its derivative.
         • input-to-hidden weights (like aw, bw, cw) via the chain rule.
        """
        error = self.__d_cost(actual)
        d_out = self.__d_sigmoid(self.output)
        # Update hidden-to-output weights.
        for i in range(self.hidden_nodes):
            grad = error * d_out * self.hidden_activations[i]
            self.weights_hidden_output[i] += self.alpha * grad
        # Update input-to-hidden weights.
        for i in range(self.hidden_nodes):
            d_hidden = self.__d_sigmoid(self.hidden_activations[i])
            for j in range(self.input_nodes):
                grad = error * d_out * self.weights_hidden_output[i] * d_hidden * self.last_input[j]
                self.weights_input_hidden[i][j] += self.alpha * grad

    def cost(self, actual):
        return (actual - self.output) ** 2

    def train(self, training_data, labels):
        """
        Trains the network by dispatching to CPU or GPU training.
        training_data is a list of samples, each a list of input features.
        labels is a corresponding list of expected outputs.
        """
        if self.use_gpu:
            self.train_gpu(training_data, labels)
        else:
            # CPU training loop.
            for ep in range(self.epoch):
                for i, sample in enumerate(training_data):
                    self.forward_prop(sample)
                    self.back_prop(labels[i])
                if ep % 1000 == 0:
                    self.forward_prop(training_data[0])
                    print("===============================")
                    print("Epoch =", ep, "Cost =", self.cost(labels[0]))

    def predict(self, inputs):
        out = self.forward_prop(inputs)
        return 1 if out > 0.5 else 0

    def predict_proba(self, inputs):
        out = self.forward_prop(inputs)
        return out if out >= 0.5 else 1 - out

    def train_gpu(self, training_data, labels):
        """
        Trains the network on the GPU using PyCUDA.
        The CUDA kernel performs forward propagation and back propagation (with atomic updating)
        for each sample. A delay is inserted at the end of each epoch (using time.sleep)
        so that GPU usage can be monitored.
        
        NOTE: This kernel assumes that self.hidden_nodes <= 128.
        """
        import time
        import pycuda.driver as cuda
        import pycuda.autoinit   # Automatically initializes the CUDA driver and context
        from pycuda.compiler import SourceModule
        import numpy as np

        num_samples = len(training_data)
        # Flatten training data.
        flat_training = np.array(training_data, dtype=np.float32).flatten()
        labels_np = np.array(labels, dtype=np.float32)
        # Flatten weight matrices.
        wih_flat = np.array([w for row in self.weights_input_hidden for w in row], dtype=np.float32)
        who_flat = np.array(self.weights_hidden_output, dtype=np.float32)

        # Allocate GPU memory.
        training_data_gpu = cuda.mem_alloc(flat_training.nbytes)
        labels_gpu = cuda.mem_alloc(labels_np.nbytes)
        wih_gpu = cuda.mem_alloc(wih_flat.nbytes)
        who_gpu = cuda.mem_alloc(who_flat.nbytes)

        # Copy data from host to GPU.
        cuda.memcpy_htod(training_data_gpu, flat_training)
        cuda.memcpy_htod(labels_gpu, labels_np)
        cuda.memcpy_htod(wih_gpu, wih_flat)
        cuda.memcpy_htod(who_gpu, who_flat)

        # CUDA kernel for training.
        mod = SourceModule("""
            #include <math.h>
            __device__ float sigmoid(float x) {
                return 1.0f / (1.0f + expf(-x));
            }
            __global__ void train_kernel(float *weights_input_hidden, float *weights_hidden_output,
                                         const float *training_data, const float *labels,
                                         int num_samples, int input_nodes, int hidden_nodes,
                                         float alpha)
            {
                int sample = blockIdx.x * blockDim.x + threadIdx.x;
                if(sample < num_samples)
                {
                    float hidden[128];  // Temporary storage for hidden activations.
                    // Forward propagation: compute hidden activations.
                    for (int i = 0; i < hidden_nodes; i++){
                        float sum = 0.0f;
                        for (int j = 0; j < input_nodes; j++){
                            sum += training_data[sample * input_nodes + j] *
                                   weights_input_hidden[i * input_nodes + j];
                        }
                        hidden[i] = sigmoid(sum);
                    }
                    // Compute output activation.
                    float out_sum = 0.0f;
                    for (int i = 0; i < hidden_nodes; i++){
                        out_sum += hidden[i] * weights_hidden_output[i];
                    }
                    float output = sigmoid(out_sum);
                    float error = 2.0f * (labels[sample] - output);
                    float d_out = output * (1.0f - output);
                    // Update hidden-to-output weights.
                    for (int i = 0; i < hidden_nodes; i++){
                        float grad = error * d_out * hidden[i];
                        atomicAdd(&weights_hidden_output[i], alpha * grad);
                    }
                    // Update input-to-hidden weights.
                    for (int i = 0; i < hidden_nodes; i++){
                        float d_hidden = hidden[i] * (1.0f - hidden[i]);
                        for (int j = 0; j < input_nodes; j++){
                            float grad = error * d_out * weights_hidden_output[i] * d_hidden *
                                         training_data[sample * input_nodes + j];
                            atomicAdd(&weights_input_hidden[i * input_nodes + j], alpha * grad);
                        }
                    }
                }
            }
        """)
        train_kernel = mod.get_function("train_kernel")
        block_size = 256
        grid_size = (num_samples + block_size - 1) // block_size

        # GPU training loop with a delay at the end of every epoch.
        for ep in range(self.epoch):
            train_kernel(wih_gpu, who_gpu, training_data_gpu, labels_gpu,
                         np.int32(num_samples), np.int32(self.input_nodes),
                         np.int32(self.hidden_nodes), np.float32(self.alpha),
                         block=(block_size, 1, 1), grid=(grid_size, 1))
            if ep % 10 == 0:
                # Copy weights back for monitoring cost.
                cuda.memcpy_dtoh(who_flat, who_gpu)
                cuda.memcpy_dtoh(wih_flat, wih_gpu)
                sample = training_data[0]
                hidden = []
                for i in range(self.hidden_nodes):
                    s = 0.0
                    for j in range(self.input_nodes):
                        s += sample[j] * wih_flat[i * self.input_nodes + j]
                    hidden.append(self.__sigmoid(s))
                out_sum = 0.0
                for i in range(self.hidden_nodes):
                    out_sum += hidden[i] * who_flat[i]
                output = self.__sigmoid(out_sum)
                cost_val = (labels[0] - output) ** 2
                print("Epoch =", ep, "Cost =", cost_val)
            # Added delay so you can monitor GPU usage.
            # import time
            # time.sleep(0.001)

        # Copy final weights from GPU back to host.
        cuda.memcpy_dtoh(wih_flat, wih_gpu)
        cuda.memcpy_dtoh(who_flat, who_gpu)
        self.weights_input_hidden = []
        for i in range(self.hidden_nodes):
            row = []
            for j in range(self.input_nodes):
                row.append(float(wih_flat[i * self.input_nodes + j]))
            self.weights_input_hidden.append(row)
        self.weights_hidden_output = [float(x) for x in who_flat]

# ---------------- Main Program Example ----------------
if __name__ == '__main__':
    # Seed for reproducibility.
    random.seed(42)
    
    # Generate 100 training examples based on the rule: label = 1 if (A - (B+C) > 0) else 0.
    all_data = []
    all_labels = []
    for i in range(100):
        a_val = random.randint(0, 20)   # A from 0 to 20.
        b_val = random.randint(0, 10)   # B from 0 to 10.
        c_val = random.randint(0, 10)   # C from 0 to 10.
        all_data.append([a_val, b_val, c_val])
        label = 1 if (a_val - (b_val + c_val) > 0) else 0
        all_labels.append(label)
    
    # Shuffle the dataset.
    combined = list(zip(all_data, all_labels))
    random.shuffle(combined)
    all_data[:], all_labels[:] = zip(*combined)
    
    # Create an 80/20 train-test split.
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    train_labels = all_labels[:split_idx]
    test_data = all_data[split_idx:]
    test_labels = all_labels[split_idx:]
    
    print("Training on", len(train_data), "samples; Testing on", len(test_data), "samples.")
    print("First 5 training examples:")
    for i in range(5):
        print("Sample:", train_data[i], "Label:", train_labels[i])
    
    # Initialize the network (set use_gpu=True to use GPU support).
    model = NeuralNetwork(input_nodes=3, hidden_nodes=3, epoch=100000, alpha=0.01, use_gpu=True)
    
    # Train the network.
    model.train(train_data, train_labels)
    
    # Evaluate on the test set.
    correct = 0
    for sample, label in zip(test_data, test_labels):
        prediction = model.predict(sample)
        if prediction == label:
            correct += 1
    accuracy = correct / len(test_data)
    print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))
    
    # Optionally, print details for each test sample.
    for sample, label in zip(test_data, test_labels):
        pred = model.predict(sample)
        prob = model.predict_proba(sample)
        print("Sample:", sample, "Predicted:", pred, "Probability:", prob, "Expected:", label)