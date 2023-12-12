import os
import sys
import Datum


class DataReader():
    def __init__(self, filePath):
        self.exampleList = []

        inputFile = open((os.path.join(sys.path[0], filePath)), "r")
        for fileLine in inputFile:
            lineValues = fileLine.split()
            if len(lineValues) >= 2:
                currentExample = Datum.Datum()
                currentExample.setType(float(lineValues.pop((len(lineValues) - 1))))
                for value in lineValues:
                    currentExample.addParameter(float(value))
                self.exampleList.append(currentExample)

        inputFile.close()

        self.trainingSet = None
        self.holdoutSet = None

    def partitionHoldoutSet(self, k, index):
        if (index > k):
            return None

        if (self.trainingSet == None):
            self.trainingSet = []
        if (self.holdoutSet == None):
            self.holdoutSet = []
        for ex in range(0, len(self.exampleList)):
            if (((ex + index) % k) != 0):
                self.trainingSet.append(self.exampleList[ex])
            else:
                self.holdoutSet.append(self.exampleList[ex])

    def partitionIntoBatches(self, batchsize):
        batchNum = int(len(self.exampleList) / batchsize)
        # print("Number of Batches: %d\n" %batchNum)
        batches = [[]] * batchNum

        for index in range(0, batchNum):
            batches[index] = self.exampleList[index::batchNum]

        # print("Current State of Batches:\n")
        # for itr in range(0, batchNum):
            # print("Batch %d :\n" %(itr))
            # for datum in batches[itr]:
            # print("%s\n" %datum.__str__())

        return batches

    def printData(self):
        for ex in range(0, len(self.exampleList)):
            print("Example %d\n" % ex)
            self.exampleList[ex].printDatum()

# dataReader = DataReader("C:\\Users\\15854\\Documents\\UR\\Year 4\\Spring 2022\\CSC 246\\CSC 246 Project 2\\dataProject2\\data\\xorSmoke")
# print("Bird")
# dataReader.printData()
