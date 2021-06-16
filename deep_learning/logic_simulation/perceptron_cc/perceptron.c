//
// Copyright © 2021 Max Xie, All rights reserved.
//
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

#define DEFAULT_NUM_INPUTS 3
#define SAMPLE_SIZE 4
#define NUM_INPUTS 3

struct Perceptron {
    uint32_t num_inputs;
    double *weights;
};

typedef struct Perceptron Perceptron;

// Create a Perceptron
Perceptron *pc_create(uint32_t num_inputs, double initial_weight) {
    Perceptron *pc = (Perceptron *)malloc(sizeof(Perceptron));
    if (pc) {
        pc->num_inputs = (num_inputs == 0) ? DEFAULT_NUM_INPUTS : num_inputs;
        uint32_t weight = initial_weight;
        pc->weights = (double *)calloc(num_inputs, sizeof(double));
        // Initialize the weights
        for (uint32_t i = 0; i < pc->num_inputs; i++) {
            pc->weights[i] = weight;
        }
    }
    return pc;
}

// Delete a Perceptron
void pc_delete(Perceptron **pc) {
    if (*pc && (*pc)->weights) {
        free((*pc)->weights);
        free(*pc);
    }
    *pc = NULL;
}

// Calculate the weighted sum and return
double weighted_sum(Perceptron *pc, uint32_t *inputs) {
    double weighted_sum = 0.0;
    for (uint32_t i = 0; i < pc->num_inputs; i++) {
        weighted_sum += pc->weights[i] * inputs[i];
    }
    return weighted_sum;
}

// Return a result between 1 and 0 using sigmoid function
double activation(int32_t z) {
    double denominator = 1.0 + exp(-z);
    double result = 1.0/denominator;
    return result;
}

// Return a classification threshold of either 0 or 1
// we use 0.5 as a threshold here for ease of understanding
uint32_t threshold(double value) {
    double threshold = 0.5;
    if (value >= threshold) {
        return 1;
    }
    return 0;
}

// Train the neural network
void training(Perceptron *pc, uint32_t training_set[SAMPLE_SIZE][NUM_INPUTS], uint32_t *expected_data, uint32_t training_data_size) {
    bool found_line = false;
    uint32_t iterations = 0;
    while (!found_line) {
        printf("iterations=%d\n", iterations);
        double total_error = 0;
        for (uint32_t i = 0; i < training_data_size; i++) {
            // Calculate the weighted_sum and pass it through activation function
            double prediction_old = activation(weighted_sum(pc, training_set[i]));
            uint32_t prediction = threshold(prediction_old);
            // Check prediction against predicted data, and calculate the loss
            uint32_t actual = expected_data[i];
            int32_t error = actual - prediction;
            total_error += abs(error);
            // adjust the weight
            for (uint32_t j = 0; j < pc->num_inputs; j++) {
                pc->weights[j] += (int32_t)(error * training_set[i][j]);
            }
        }
        printf("[error] %f\n", total_error);
        // End the training process when the total error is 0
        if (total_error < 0.01) {
            found_line = true;
            printf("Training completed! Total Error=%.5f\n", total_error);
        }
        iterations++;
    }
}

// Return the prediction and confidence for a given testing data set
void predict(Perceptron *pc, uint32_t *testing_set) {
    double final_prob = 0.0;
    // Calculate the weighted_sum and pass it through activation function
    double probability = activation(weighted_sum(pc, testing_set));
    uint32_t prediction = threshold(probability);

    if (prediction == 1) {
        final_prob = probability;
    } else {
        final_prob = 1 - probability;
    }
    printf("Prediction: %d\n", prediction);
    printf("probability: %.5f\n", final_prob);
}

int main() {
    // Create a Perceptron
    Perceptron *pc = pc_create(NUM_INPUTS, 1.0);
    // Create training data
    // Use the prediction result of (A - (B + C) > 0) to build the dataset
    uint32_t small_training_set[SAMPLE_SIZE][NUM_INPUTS] = {
        { 1, 1, 0 },
        { 2, 0, 1 },
        { 3, 2, 0 },
        { 4, 5, 1 }
    };
    // Create expected result
    uint32_t expected_result[SAMPLE_SIZE] = { 0, 1, 1, 0 };

    // Train the model
    training(pc, small_training_set, expected_result, SAMPLE_SIZE);

    // Test the model, true case
    uint32_t small_testing_data[NUM_INPUTS] = { 3, 1, 1 };
    printf("Start prediction, testing_set=");
    for (int i = 0; i < NUM_INPUTS; i++) {
        printf("%d ", small_testing_data[i]);
    }
    printf("\n");

    uint32_t expected = 1;
    predict(pc, small_testing_data);
    printf("Expected: %d\n", expected);

    // Test the model, false case
    uint32_t small_testing_data2[NUM_INPUTS] = { 3, 5, 6 };
    printf("Start prediction, testing_set=");
    for (int i = 0; i < NUM_INPUTS; i++) {
        printf("%d ", small_testing_data2[i]);
    }
    printf("\n");

    uint32_t expected2 = 0;
    predict(pc, small_testing_data2);
    printf("Expected: %d\n", expected2);

    // garbage collection
    pc_delete(&pc);
}
