
class Datum:
    def __init__(self):
        self.type = []
        self.parameters = []

    def setType(self, newType):
        if (len(self.type) == 0):
            self.type.append(newType)
            self.type.append(1 - newType)

        else:
            self.type = []
            self.type.append(newType)
            self.type.append(1 - newType)

    def getType(self, index):
        return self.type[index]

    def getParameterCount(self):
        return len(self.parameters)

    def getParameter(self, index):
        return self.parameters[index]

    def getParameters(self):
        return self.parameters

    def addParameter(self, newParameter):
        self.parameters.append(newParameter)

    def setParameter(self, index, newParameter):
        self.parameters[index] = newParameter

    def setParameters(self, newParameters):
        self.parameters = newParameters

    def printDatum(self):
        for num in range(0, len(self.parameters)):
            print("x %d : %f\n" %(num, self.parameters[num]))
        print("t : %d\n" %self.type[0])

    def __str__(self):
        refString = ""
        for num in range(0, len(self.parameters)):
            refString += "x {0} : {1}\n".format(num, self.parameters[num])
        refString += "t : {0}".format(self.type[0])
        return refString
