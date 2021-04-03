# A self-coded perceptron
class Perceptron:
  def __init__(self, num_inputs=3, weights=[1,1,1]):
    self.num_inputs = num_inputs
    self.weights = weights

  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum

  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1

  def training(self, training_set):
    foundLine = False
    # Keep track of the total amount of iterations
    iterations = 0
    while not foundLine:
      print(f"Start Training... Iterations={iterations}")
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        print(f"[Predicted]{prediction}")
        actual = training_set[inputs]
        print(f"[Actual]{actual}")
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
        print(f"[Weights]{self.weights}")
      if total_error == 0:
        foundLine = True
        print(f"Training completed! Total Error={total_error}")
      iterations += 1

  def predict(self, testing_set):
    prediction = self.activation(self.weighted_sum(testing_set))
    return prediction

if __name__ == '__main__':
    cool_perceptron = Perceptron()
    # Use the prediction result of (A - (B + C) > 0) to build the dataset
    # So that we can actually check the model
    small_training_set = {(1,1,0):-1, (2,0,1):1, (3,2,0):1, (4,5,1):-1}
    cool_perceptron.training(small_training_set)
    
    # Testing the model
    testing_set = (3,1,1)
    print(f"Start Prediction, testing_set={testing_set}")
    expected = 1
    result = cool_perceptron.predict(testing_set)
    print(f"result:{result}")
    print(f"expected:{expected}")
    print(f"error={result-expected}")
