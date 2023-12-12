import InputNode

class Connection:
    def __init__(self, inNode, outNode, weight):
        self.inNode = inNode
        self.outNode = outNode
        self.weight = weight

    def setInNode(self, newInNode):
        self.inNode = newInNode

    def getInNode(self):
        return self.inNode

    def setOutNode(self, newOutNode):
        self.outNode = newOutNode 
    
    def getOutNode(self):
        return self.outNode

    def setWeight(self, newWeight):
        self.weight = newWeight

    def getWeight(self):
        return self.weight

    def getOutgoingConnection(self):
        return self.weight * self.inNode.getActivation()

    def printConnection(self):
        print(self.weight)