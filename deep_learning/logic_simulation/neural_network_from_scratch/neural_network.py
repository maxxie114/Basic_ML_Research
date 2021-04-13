# Neural Network Coded entirely from scratch without any third party libraries
# Attention: This project only work with data sets that have three features
import math
class NeuralNetwork:
    # Ignore bias for easy implementation
    def __init__(self, alpha=0.0001, epoch=1000):
        self.alpha = alpha
        self.epoch = epoch
        # Initialize all the weights and layers nodes
        # Refer to the pdf for more info
        self.xn, self.yn, self.zn, self.outn = 0,0,0,0
        self.xw, self.yw, self.zw = 0,0,0
        self.aw1, self.aw2, self.aw3 = 0,0,0
        self.bw1, self.bw2, self.bw3 = 0,0,0
        self.cw1, self.cw2, self.cw3 = 0,0,0
        self.a, self.b, self.c = 0,0,0
        
    def __sigmoid(self, z):
        """Return the result of the sigmoid function"""
        return 1 / (1 + math.exp(-z))

    def __d_sigmoid(self, z):
        """Return the derivative of the sigmoid function"""
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))

    def __cost(self, actual, predicted):
        """Calculate and return the cost using squared error"""
        return (actual - predicted) ** 2

    def __d_cost(self, actual, predicted):
        """Calculate and return the derivative of the cost function"""
        return 2 * (actual - predicted)

    def forward_prop(self,a,b,c):
        """Run the entire forward propagation process of
           a neural network

           Description:
                The algorithm of forward prop is that, a sum of all the input multiplied
                by all weights are passed through an activation function to get another 
                layer of input, and this new layer of input is multiplied by another
                layer of weights, and passed through another layer of activation function
                this process repeat until the result finally reached the output layer.

           Args:
                a: The first input to the neural network
                b: The second input to the neural network
                c: the third input to the neural network
        """
        # The sum of all the input in the input layer
        # multiplied by all the weights
        x = a* self.aw1 + b * self.bw1 + c*self.cw1
        y = a* self.aw2 + b * self.bw2 + c*self.cw2
        z = a* self.aw3 + b * self.bw3 + c*self.cw3
        # all the sum are passed through an activation function
        self.xn = self.__sigmoid(x)
        self.yn = self.__sigmoid(y)
        self.zn = self.__sigmoid(z)
        # The sum of all the input from the first layer multiplied
        # by the weights to the second layer, or the output layer
        out = x*self.xw + y*self.yw + z*self.zw
        # Result are passed through an activation function
        # At the output layer
        self.outn = self.__sigmoid(out)
        self.a = a
        self.b = b
        self.c = c

    def back_prop(self, actual):
        """Run the entire back propagation process and improve the weights using gradient descent
           
           Description:
                The algorithm of back propagation is that a cost is calculated, or the distance between
                the predicted value and the actual output, and this cost is passed back through the
                activation function to adjust every single weights, and by subtracting the cost times a 
                learning rate, and weights can be improved to be able to reduce the cost, this is also
                called gradient descent algorithm

           Equations:
                new_weight = old_weight - learning_rate * derivative_of_cost_with_respect_to_weight
                derivative_of_cost_with_respect_to_weights = (derivative_of_cost * derivative_of_activation_func * input)
                hidden_layer_deri_cost_respect_to_weights = (derivative_of_cost_output_layer * derivative_of_activation_output_layer * old_weight_output_layer)
                
           Args:
                actual: the actual label of the data
        """
        # Adjusting the weights of the output layer using gradient descent
        self.xw += (self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.xn) * self.alpha
        self.yw += (self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.yn) * self.alpha
        self.zw += (self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.zn) * self.alpha
        # Adjusting the weight for the hidden layer using gradient descent
        self.aw1 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.xw) * self.__d_sigmoid(self.xn) * self.a) * self.alpha
        self.bw1 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.xw) * self.__d_sigmoid(self.xn) * self.b) * self.alpha
        self.cw1 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.xw) * self.__d_sigmoid(self.xn) * self.c) * self.alpha
        
        self.aw2 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.yw) * self.__d_sigmoid(self.yn) * self.a) * self.alpha
        self.bw2 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.yw) * self.__d_sigmoid(self.yn) * self.b) * self.alpha
        self.cw2 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.yw) * self.__d_sigmoid(self.yn) * self.c) * self.alpha
        
        self.aw3 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.zw) * self.__d_sigmoid(self.zn) * self.a) * self.alpha
        self.bw3 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.zw) * self.__d_sigmoid(self.zn) * self.b) * self.alpha
        self.cw3 += ((self.__d_cost(actual,self.outn) * self.__d_sigmoid(self.outn) * self.zw) * self.__d_sigmoid(self.zn) * self.c) * self.alpha
        # print(f"{self.aw1},{self.aw2},{self.aw3},{self.bw1},{self.bw2},{self.bw3},{self.cw1},{self.cw2},{self.cw3},{self.xw},{self.yw},{self.zw}")
    
    def train(self,a,b,c,actual):
        """Run the training loop over a predefined epoch

           Args:
                a: The first feature set
                b: The second feature set
                c: The third feature set
                actual: A set of final prediction
        """
        for i in range(self.epoch):
            for j in range(len(a)):
                self.forward_prop(a[j],b[j],c[j])
                self.back_prop(actual[j])
                if i % 100 == 0:
                    print("===============================")
                    print(f"EPOCH={i}")
                    print(f"COST={self.__cost(actual[j],self.outn)}")

    def __threshold_classification(self,predicted):
        """Convert a probability value to an actual prediction of 0 or 1"""
        if predicted > 0.5:
            return 1
        else:
            return 0
    
    def predict(self,a,b,c):
        """Run the prediction from pretrained weights, and run it through threshold classification"""
        self.forward_prop(a,b,c)
        return self.__threshold_classification(self.outn)

    def predict_proba(self,a,b,c):
        """Run the prediction from pretrained weights, and directly return the probability"""
        self.forward_prop(a,b,c)
        if self.outn < 0.5:
            return 1 - self.outn
        return self.outn
           

if __name__ == '__main__':
    model = NeuralNetwork(epoch=100000,alpha=0.01)
    
    # Create dataset
    # Use the prediction result of (A - (B + C) > 0) to build the dataset
    # So that we can actually check the model
    # small_training_set = {(1,1,0):0, (2,0,1):1, (3,2,0):1, (4,5,1):0}
    a = [1,2,3,4]
    b = [1,0,2,5]
    c = [0,1,0,1]
    actual_result = [0,1,1,0]

    model.train(a,b,c,actual_result)
    
    # testing the model
    x_test = (3,1,1)
    print(f"Start Prediction, testing_set={x_test}")
    expected = 1
    result = model.predict(x_test[0],x_test[1],x_test[2])
    print(f"result:{result}\tprobabilities:{model.predict_proba(x_test[0],x_test[1],x_test[2])}")
    print(f"expected:{expected}")
    print(f"error={expected-result}")
    
    # testing the model
    x_test2 = (10,9,2)
    print(f"Start Prediction, testing_set={x_test2}")
    expected2 = 0
    result2 = model.predict(x_test2[0],x_test2[1],x_test2[2])
    print(f"result:{result2}\tprobabilities:{model.predict_proba(x_test2[0],x_test2[1],x_test2[2])}")
    print(f"expected:{expected2}")
    print(f"error={expected2-result2}")
