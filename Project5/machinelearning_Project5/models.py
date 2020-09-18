import nn
import math
import numpy as np

class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new LogisticRegressionModel instance.

        A logistic regressor classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        Initialize self.w and self.alpha here
        self.alpha = *some number* (optional)
        self.w = []
        """
        "*** YOUR CODE HERE ***"
        self.alpha = 0.42
        #4.5 - 99.6, 4.6 - 99.4
        weights = []
        for i in range(dimensions):
            weights.append(1)
        self.w = weights

    def get_weights(self):
        """
        Return a list of weights with the current weights of the regression.
        """

        return self.w
        

    def DotProduct(self, w, x):
        """
        Computes the dot product of two lists
        Returns a single number
        """
        "*** YOUR CODE HERE ***"

        s = 0
        s = np.dot(w, x)
        return s

    def sigmoid(self, x):
        """
        compute the logistic function of the input x (some number)
        returns a single number
        """
        "*** YOUR CODE HERE ***"
        return 1/(1 + math.exp(x * -1))


    def run(self, x):
        """
        Calculates the probability assigned by the logistic regression to a data point x.

        Inputs:
            x: a list with shape (1 x dimensions)
        Returns: a single number (the probability)
        """
        "*** YOUR CODE HERE ***"
        return self.sigmoid(self.DotProduct(self.w, x))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        p = self.run(x)
        if p >= 0.5:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the logistic regression until convergence (this will require at least two loops).
        Use the following internal loop stucture

        for x,y in dataset.iterate_once(1):
            x = nn.as_vector(x)
            y = nn.as_scalar(y)
            ...

        """
        "*** YOUR CODE HERE ***"
        while 1:
            correct = True
            for x, y in dataset.iterate_once(1):
                
                xvec = nn.as_vector(x)
                yvec = nn.as_scalar(y)
                c = self.get_prediction(xvec)
                p = self.run(xvec)

                if yvec != c:
                    correct = False
                    cpw = self.w.copy()
                    for i in range(len(self.w)):
                        self.w[i] = cpw[i] + self.alpha * (yvec - p) * p * (1-p) * xvec[i]
            if correct:
                break


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        We did this for you. Nothing for you to do here.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        p = nn.as_scalar(self.run(x))
        if p >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"    
        while(1):
            correct = True
            for x, y in dataset.iterate_once(1):                 
                yvec = nn.as_scalar(y)
                c = self.get_prediction(x)

                if yvec != c:
                    correct = False
                    nn.Parameter.update(self.w, x, yvec)
            if correct:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, dimensions):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.alpha = 0.003
        self.weight = nn.Parameter(1, dimensions)
        self.bias = nn.Parameter(1, 1)
    
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        
        ltranNodes = nn.Linear(x, self.weight)
        output = nn.AddBias(ltranNodes, self.bias)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x,y)
                gradients = nn.gradients(loss, [self.weight, self.bias])
                self.weight.update(gradients[0], -self.alpha)
                self.bias.update(gradients[1], -self.alpha)
                
            print(nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.15:
                return



    def closedFormSolution(self, X, Y):
        """
        Compute the closed form solution for the 2D case
        Input: X,Y are lists
        Output: b0 and b1 where y = b1*x + b0
        """
        "*** YOUR CODE HERE ***"
        print("X: {}".format(X))
        print("Y: {}".format(Y))

        n = len(X)
        xsum = 0
        ysum = 0
        xysum = 0
        xxsum = 0

        for i in range(len(X)):
            xsum += X[i]
            xysum += X[i]*Y[i]
            xxsum += X[i]*X[i]
        for i in range(len(Y)):
            ysum += Y[i]

        b1 = (n*(xysum) - xsum*ysum)/(n*xxsum - xsum*xsum)
        b0 = (ysum - b1*xsum)/n

        return b0, b1

class PolyRegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, order):
        # Initialize your model parameters here
        """
        initialize the order of the polynomial, as well as two parameter nodes for weights and bias
        """
        "*** YOUR CODE HERE ***"
        self.order = order
        self.weight = nn.Parameter(1, 1)
        self.bias = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

       
        ltranNodes = nn.Linear(x, self.weight)
        output = nn.AddBias(ltranNodes, self.bias)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)
       

    def computePolyFeatures(self, point):
        """
        Compute the polynomial features you need from the input x
        NOTE: you will need to unpack the x since it is wrapped in an object
        thus, use the following function call to get the contents of x as a list:
        point_list = nn.as_vector(point)
        Once you do that, create a list of the form (for batch size of n): [[x11, x12, ...], [x21, x22, ...], ..., [xn1, xn2, ...]]
        Once this is done, then use the following code to convert it back into the object
        nn.Constant(nn.list_to_arr(new_point_list))
        Input: a node with shape (batch_size x 1)
        Output: an nn.Constant object with shape (batch_size x n) where n is the number of features generated from point (input)
        """
        "*** YOUR CODE HERE ***"
        point_list = nn.as_vector(point)
        new_point_list = []
        for i in range(len(point_list)):
            poly = []
            for n in range(self.order):
                poly.append(point_list[i]**n)
            new_point_list.append(poly)        

        return nn.Constant(np.array(new_point_list))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(self.computePolyFeatures(x),y)
                gradients = nn.gradients(loss, [self.weight, self.bias])
                self.weight.update(gradients[0], -self.alpha)
                self.bias.update(gradients[1], -self.alpha)
                
            print(nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.15:
                return

class FashionClassificationModel(object):
    """
    A model for fashion clothing classification using the MNIST dataset.

    Each clothing item is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.weight = nn.Parameter(784, 10)
        self.bias = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        ltranNodes = nn.Linear(x, self.weight)
        output = nn.AddBias(ltranNodes, self.bias)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        while 1:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
   
               
            