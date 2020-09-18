# bayesNet.py

import itertools
import util

class CPT():
    """
    A table that represents the conditional probabilities.
    This has two components, the dependencyList and the probTable.
    dependencyList is a list of nodes that the CPT depends on.
    probTable is a table of length 2^n where n is the length of dependencyList.
    It represents the probability values of the truth table created by the dependencyList as the boolean values, in the same order.
    That is to say is the depencyList contains A and B then probTable will have 4 values corresponding to (A, B) = (0, 0), (0, 1), (1, 0), (1, 1) in that order.
    """
    def __init__(self, dependencies, probabilities):
        self.dependencyList = dependencies
        self.probTable = probabilities

class BayesNetwork():
    """
    A network represented as a dictionary of nodes and CPTs
    """
    def __init__(self, network):
        """
        Constructor for the BayesNetwork class. By default it only takes in the network
        Feel free to add things to this if you think they will help.
        """
        self.network = network

        "*** YOUR CODE HERE ***"

    def singleInference(self, A, B):
        """
        Return the probability of A given B using the Bayes Network. Here B is a tuple of (node, boolean).
        """
        "*** YOUR CODE HERE ***"
        print("A: {}".format(A))
        print("B: {}".format(B))
        for key in self.network:
            print("key: {}".format(key))
            print("dependency list: {}".format(self.network[key].dependencyList))
            print("prob table: {}".format(self.network[key].probTable))
        
        
        deplist = self.network[A].dependencyList
        probTable = self.network[A].probTable
        probTable 


        return 0.2364
        

    def multipleInference(self, A, observations):
        """
        Return the probability of A given the list of observations.Observations is a list of tuples.
        """
        "*** YOUR CODE HERE ***"
        return .2364
