# A self-coded perceptron (single-neuron neural net)
import numpy as np
class Perceptron:
  def __init__(self, num_inputs=3, weights=[1,1,1]):
    # num_inputs is the number of inputs that will be included in a list of data
    self.num_inputs = num_inputs
    # weights are the weight assigned to each synapse for the neural network
    self.weights = weights

  def weighted_sum(self, inputs):
    """return the sum for all inputs multiplied by its weight"""
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum

  def activation(self, z):
    """Return a result between 1 and 0 using a sigmoid function"""
    denominator = 1 + np.exp(-z)
    result = 1/denominator
    return result

  def threshold(self, value):
    """Return a classification threshold of either 0 or 1
       we use 0.5 as a threshold here for ease of understanding
    """
    threshold = 0.5
    if value >= threshold:
        return 1
    else:
        return 0
 
  def training(self, training_set):
    """Train the neural network"""
    # Start foundline with False until the neural network is able to 
    # Correctly come up with a line that can classify the dataset
    foundLine = False
    # Keep track of the total amount of iterations
    iterations = 0
    while not foundLine:
      print(f"Start Training... Iterations={iterations}")
      total_error = 0
      for inputs in training_set:
        # Start with random weights 1,1,1
        # And uses simple multiplication and addition to get a weighted sum
        # pass it through the activation function to get either a 1 or 0
        prediction_old = self.activation(self.weighted_sum(inputs))
        # pass the prediction through the threshold to be get a value of 0 or 1
        # for easy training
        prediction = self.threshold(prediction_old)
        print(f"[Predicted]{prediction}")
        actual = training_set[inputs]
        print(f"[Actual]{actual}")
        # Calculate the error by subtracting actual with predicted value
        # So error can either be 2 or -2
        # each piece of data in the dataset will have an error rate
        error = actual - prediction
        # total error is the sum of all errors in the entire dataset
        # Take the absolute value so that error can only be 2, so both
        # -2 and +2 will have to same effect to the neural network
        total_error += abs(error)
        for i in range(self.num_inputs):
          # modify the weight by using error multiply input, this will result in a
          # better weight that can work better with the dataset
          # This is back backpropagation
          self.weights[i] += error*inputs[i]
        print(f"[Weights]{self.weights}")
      # End the training process when the total error is 0
      if total_error == 0:
        foundLine = True
        print(f"Training completed! Total Error={total_error}")
      iterations += 1

  def predict(self, testing_set):
    """return the prediction and confidence for a given testing data"""
    # Run prediction by multiply inputs with the weight and map it
    # Through the activation function
    final_prob = 0
    probability = self.activation(self.weighted_sum(testing_set))
    prediction = self.threshold(probability)
    if prediction == 1:
        final_prob = probability
    else:
        final_prob = 1 - probability
    return [prediction, final_prob]

if __name__ == '__main__':
    cool_perceptron = Perceptron()
    # Use the prediction result of (A - (B + C) > 0) to build the dataset
    # So that we can actually check the model
    small_training_set = {(1,1,0):0, (2,0,1):1, (3,2,0):1, (4,5,1):0}
    cool_perceptron.training(small_training_set)
    
    # Testing the model
    testing_set = (3,1,1)
    print(f"Start Prediction, testing_set={testing_set}")
    expected = 1
    result = cool_perceptron.predict(testing_set)
    print(f"result:{result[0]}\tprobability:{result[1]}")
    print(f"expected:{expected}")
    print(f"error={result[0]-expected}")
