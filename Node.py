
class Node():

    def __init__(self):
        self.activation = 0.0
        self.outgoingConnections = None
        self.incomingConnections = None 

    def getActivation(self):
        return self.activation

    def setActivation(self, num):
        self.activation = num 

    def addOutgoingConnection(self, connection):
        if (self.outgoingConnections != None):
            self.outgoingConnections.append(connection)

    def addOutgoingConnections(self, connections):
        if (self.outgoingConnections != None):
            for connection in connections:
                self.outgoingConnections.append(connection)

    def setOutgoingConnections(self, connections):
        self.outgoingConnections = connections

    def getOutgoingConnections(self):
        return self.outgoingConnections

    def addIncomingConnection(self, connection):
        if (self.incomingConnections != None):
            self.incomingConnections.append(connection)

    def addIncomingConnections(self, connections):
        if (self.incomingConnections != None):
            for connection in connections:
                self.incomingConnections.append(connection)
    
    def setIncomingConenctions(self, connections):
        self.incomingConnections = connections

    def getIncomingConnections(self):
        return self.incomingConnections

    def derivative(self):
        return 0.0

    def printNode(self):
        print("Activation: %f\n" %self.activation)