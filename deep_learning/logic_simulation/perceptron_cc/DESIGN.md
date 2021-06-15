# Single-layer Percetron Algorithm
- This is an implementation of the single-layer perceptron algorithm

# The algorithm
- The algorithm have two parts, forward propagation, and back propagation, the program works by looping through forward prop, get an output, calculate a loss using a certain loss function, and than use backward prop to adjust the values of all the weights until the error rate is lower than a certain number.

### Forward propagation
- This is the process of combining all the inputs into a value between 0 and 1
    - 1. Assign a random weight to each of the input nodes
    - 2. Multiply each input value by their corresponding weight to get a weighted input
    - 3. At all the weighted input together to get a weighted sum
    - 4. Pass the weighted sum into an activation function, we are using sigmoid function here, the point of the activation function is to map the weighted sum into a value between a specific range, in this case, it is to map it between 0 and 1. 
    - 5. The output of the activation function is the final prediction
### Backward propagation
- This is the process of adjusting the weights in order to get the best prediction
    - 1. Take the output from forward prop
    - 2. Calculate a loss
    - 3. Run it through an algorithm like stochastic gradient descent in order to adjust all the weights
    - 4. Run forward prop again with the adjusted weights
    - 5. Repeat the process until a certain amount of iterations(EPOCH), or until the error rate is below a certain acceptable value
![An image of the single perceptron neural network](https://pythonmachinelearning.pro/wp-content/uploads/2017/09/Single-Perceptron.png.webp)

# Define
- Here are the macro that are defined
```cpp
#define DEFAULT_NUM_INPUTS 3
#define SAMPLE_SIZE 4
#define NUM_INPUTS 3
```

# Perceptron
- This struct acts like a class, and it store all the weights of the neural network
```cpp
struct Perceptron {
    uint32_t num_inputs;
    double *weights;
};
```

### pc create
- Create a Perceptron
```cpp
Perceptron *pc_create(uint32_t num_inputs, double initial_weight) {
    pc = allocate memory of size Perceptron
    if pc exist {
        if num_inputs is 0, set the number of inputs to default num_input, else, let the num_input to be the provided num_inputs
        for i from 0 to num_inputs {
            set weight at index i to be the initial_weight
        }
    }
    return pc
}
```
### pc delete
- Delete the Perceptron and free all the memory of the struct
```cpp
void pc_delete(Perceptron **pc) {
    if pc and the weights array exist {
        free the weight array
        free the pc pointer
    }
    set the pc pointer to NULL
}
```
### weighted sum
- Calculate the weighted sum of the entire input set
```cpp
double weighted_sum(Perceptron *pc, uint32_t *inputs) {
    set the weighted_sum to 0
    for i from 0 to num_inputs {
        multiply the weight at index i of the weight array by the input at index i of the input set and add the result to weighted_sum
    }
    return the weighted sum
}
```
### activation
- This is the activation function, in this function we use sigmoid function to map the weighted sum between 0 and 1 so we can get a correct boolean output
```cpp
double activation(int32_t z) {
    donominator = 1 + e^(-z)
    result = 1 / denominator
    return the result
}
```
### threshold
- map the output to a number that is either 1 or 0, so the final output is boolean
```cpp
uint32_t threshold(double value) {
    threshold = 0.5
    if the value is bigger than the threshold {
        return 1
    } else {
        return 0
    }
}
```
### training
- This is the training loop, it contains both the forward prop and the backward prop
```cpp
void training(Perceptron *pc, uint32_t training_set[SAMPLE_SIZE][NUM_INPUTS], 
  uint32_t *expected_data, uint32_t training_data_size) {
    // A neural network is like a function, that is trying to draw a line of best fit 
    // between the input and the expected output
    found_line = false
    iterations = 0
    while the line of best fit is not found {
        total_error = 0
        for i from 0 to training_data_size {
            // forward prop
            prediction_old = the output of activation function with the weighted sum of the index i of training_set as the input
            prediction = the output of the threshold function with prediction_old as the input
            // Calculate the loss that can be used in backward prop
            actual = index i of the expected_data array
            error = actual - prediction
            total_error += the absolute value of the error
            // backward prop
            for j from 0 to num_inputs {
                the weights at index j += error multiply by index [i][j] of the training_set 
            }
        }
        if the total error < 0.01 {
            // We have reached out expected accuracy
            set found_line to true
        }
        iterations++
    }
}
```
### predict
- Run the forward prop with the testing data set, using the already adjusted weights
```cpp
void predict(Perceptron *pc, uint32_t *testing_set) {
    final_prob = 0
    probability = the output of activation function with the weighted sum of the entire testing_set
    prediction = the output of the threshold function with probability as the input

    if prediction is 1 {
        final_prob = probability
    } else {
        final_prob = 1 - probability
    }
    print both the prediction and the probability
}
```
