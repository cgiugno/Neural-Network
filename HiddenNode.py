import numpy as np
import Node 
import Connection 

class HiddenNode(Node.Node):
    def __init__(self, isSoftMax):
        Node.Node.__init__(self)
        self.incomingConnections = []
        self.outgoingConnections = []
        self.isSoftMax = isSoftMax

    def addOutgoingConnection(self, connection):
        self.outgoingConnections.append(connection)

    def addOutgoingConnections(self, connections):
        for connection in connections:
            self.outgoingConnections.append(connection)

    def getOutgoingConnections(self):
        return self.outgoingConnections

    def addIncomingConnection(self, connection):
        self.incomingConnections.append(connection)

    def addIncomingConnections(self, connections):
        for connection in connections:
            self.incomingConnections.append(connection)

    def getIncomingConnections(self):
        return self.incomingConnections

    def findActivation(self):
        # print("Finding Activation...\n")
        if (len(self.incomingConnections) == 0):
            return float('-inf')
        else:
            sum = 0.0
            for incomingConnection in self.incomingConnections:
                sum += incomingConnection.getOutgoingConnection()
                
                # print("Incoming Connection: %f\nCurrent sum: %f\n" %(incomingConnection.getOutgoingConnection(), sum))
            if (self.isSoftMax):
                self.activation = np.exp(sum)
                return np.exp(sum) # And then you divide by the sum later?
            else:
                self.activation = np.tanh(sum)
                # print("Tanh: %f\n" %np.tanh(sum))
                return np.tanh(sum)

    def getActivation(self):
        if (self.activation == 0):
            self.findActivation()
            return self.activation
        else:
            return self.activation
    
    def derivative(self):
        if (self.isSoftMax):
            pass
        else:
            return (1 - (self.activation ** 2))

    def printNode(self):
        print("Activation: %f\n" %self.activation)
        print("Outgoing Connections: ")
        for connection in self.outgoingConnections:
            connection.printConnection()
        print("Incoming Connections: ")
        for connection in self.incomingConnections:
            connection.printConnection()
