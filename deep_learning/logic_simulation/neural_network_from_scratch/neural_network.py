# Neural Network Coded entirely from scratch without any third party libraries
# Attention: This project only work with data sets that have three features
import math
import random
class NeuralNetwork:
    # Ignore bias for easy implementation
    def __init__(self, alpha=0.0001, epoch=10):
        self.alpha = alpha
        self.epoch = epoch
        # Initialize all the weights and layers nodes
        self.xn, self.yn, self.zn, self.outn = 0,0,0,0
        # 👇 one-line random initialization
        self.xw, self.yw, self.zw, \
        self.aw1, self.aw2, self.aw3, \
        self.bw1, self.bw2, self.bw3, \
        self.cw1, self.cw2, self.cw3 = [random.uniform(-1, 1) for _ in range(12)]
        self.a, self.b, self.c = 0,0,0

    def __sigmoid(self, z):
        """Return the result of the sigmoid function"""
        # stable sigmoid (no overflow)
        if z >= 0:
            t = math.exp(-z)
            return 1 / (1 + t)
        else:
            t = math.exp(z)
            return t / (1 + t)

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
        out = self.xn*self.xw + self.yn*self.yw + self.zn*self.zw
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
                epoch_cost = 0.0
                for j in range(len(a)):
                    self.forward_prop(a[j], b[j], c[j])
                    epoch_cost += self.__cost(actual[j], self.outn)
                    self.back_prop(actual[j])
                if (i + 1) % 1 == 0:  # print every 1 epochs
                    avg_cost = epoch_cost / len(a)
                    print("===============================")
                    print(f"EPOCH={i+1}")
                    print(f"AVG COST={avg_cost}")

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

    def predict_proba(self, a,b,c):
        """Run the prediction from pretrained weights, and directly return the probability"""
        self.forward_prop(a,b,c)
        return self.outn    # prob(class=1)


import random

if __name__ == '__main__':
    random.seed(123456)  # reproducible

    model = NeuralNetwork(epoch=5, alpha=0.1)

    # ---- Generate a larger synthetic dataset ----
    # Rule: label = 1 if (A - (B + C) > 0) else 0
    N = 1000  # total samples
    A_min, A_max = -20, 20
    B_min, B_max = -20, 20
    C_min, C_max = -20, 20

    A_all, B_all, C_all, Y_all = [], [], [], []
    for _ in range(N):
        a = random.randint(A_min, A_max)
        b = random.randint(B_min, B_max)
        c = random.randint(C_min, C_max)
        y = 1 if (a - (b + c) > 0) else 0
        A_all.append(a); B_all.append(b); C_all.append(c); Y_all.append(y)

    # ---- Train / test split (80/20) ----
    idx = list(range(N))
    random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    a_train = [A_all[i] for i in train_idx]
    b_train = [B_all[i] for i in train_idx]
    c_train = [C_all[i] for i in train_idx]
    y_train = [Y_all[i] for i in train_idx]

    a_test  = [A_all[i] for i in test_idx]
    b_test  = [B_all[i] for i in test_idx]
    c_test  = [C_all[i] for i in test_idx]
    y_test  = [Y_all[i] for i in test_idx]

    # ---- Train ----
    model.train(a_train, b_train, c_train, y_train)

    # ---- Evaluate on test set ----
    correct = 0
    for a, b, c, y_true in zip(a_test, b_test, c_test, y_test):
        y_pred = model.predict(a, b, c)
        correct += int(y_pred == y_true)
    accuracy = correct / len(y_test)

    print("\n=== Test Evaluation ===")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.3f}")

    # Show a few random test predictions
    print("\nSample predictions:")
    for i in random.sample(range(len(y_test)), k=min(10, len(y_test))):
        a, b, c = a_test[i], b_test[i], c_test[i]
        prob = model.predict_proba(a, b, c)
        pred = 1 if prob > 0.5 else 0
        print(f"(A,B,C)=({a},{b},{c})  prob={prob:.3f}  pred={pred}  true={y_test[i]}")

