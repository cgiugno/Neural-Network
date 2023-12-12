import Node
import Connection
import numpy as np

class OutputNode(Node.Node):
    def __init__(self, isSoftMax):
        Node.Node.__init__(self)
        self.incomingConnections = []
        self.isSoftMax = isSoftMax

    def addIncomingConnection(self, connection):
        self.incomingConnections.append(connection)

    def addIncomingConnections(self, connections):
        for connection in connections:
            self.incomingConnections.append(connection)

    def getIncomingConnections(self):
        return self.incomingConnections

    def findActivation(self):
        if (len(self.incomingConnections) == 0):
            return float('-inf')
        else:
            sum = 0.0
            for incomingConnection in self.incomingConnections:
                sum += incomingConnection.getOutgoingConnection()
            # print("Sum: %f\n" %sum)
            if (self.isSoftMax):
                self.activation = np.exp(sum)
                return np.exp(sum) #divide by denominator in NN
            else:
                self.activation = np.tanh(sum)
                return np.tanh(sum)

    def findIncomingSum(self):
        if (len(self.incomingConnections) == 0):
            return float('-inf')
        else:
            sum = 0.0
            for incomingConnection in self.incomingConnections:
                sum += incomingConnection.getOutgoingConnection()
        return sum

    def getActivation(self):
        if (self.activation == 0):
            self.findActivation()
            return self.activation
        else:
            return self.activation
    
    def derivative(self, softmaxActivations):
        softMaxDerivatives = [0.0] * len(softmaxActivations)
        for activation in range(0, len(softmaxActivations)):
            yk = self.activation
            yj = softmaxActivations[activation]

            if(yk == yj):
                softMaxDerivatives[activation] = yk * (1-yj)

            else:
                softMaxDerivatives[activation] = yk * (-yj)

        return softMaxDerivatives

    

    def printNode(self):
        print("Activation: %f\n" %self.getActivation())
        print("Incoming Connections: \n")
        for connection in self.incomingConnections:
            connection.printConnection()
    