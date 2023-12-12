import Node 
import Connection 

class InputNode(Node.Node):
    def __init__(self, inputActivation):
        Node.Node.__init__(self)
        self.activation = inputActivation
        self.outgoingConnections = [] 

    def addOutgoingConnection(self, connection):
        self.outgoingConnections.append(connection)

    def addOutgoingConnections(self, connections):
        for connection in connections:
            self.outgoingConnections.append(connection)

    def getOutgoingConnections(self):
        return self.outgoingConnections

    def getActivation(self):
        return self.activation
    
    def printNode(self):
        print("Activation: %f\n" %self.activation)
        print("Outgoing Connections: \n")
        for connection in self.outgoingConnections:
            connection.printConnection()
