import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import Node
import InputNode
import OutputNode
import HiddenNode
import Connection
import random as rand
import Datum
import DataReader


# Neural Network Class with
# inputLayer, a list of InputNodes
# outLayer, a list of OutputNodes
# hiddenLayer, a list of lists of Hidden Nodes (representing each layer of hidden nodes in the neural network)
# delta, a list of lists of errors for each node in the neural network
# activation, a list of lists of activations for each node in the neural network 
class NeuralNetwork:
    def __init__(self, inputNum, outputNum, hiddenNums):
        self.inputLayer = []
        self.outLayer = []
        self.hiddenLayer = [[]] * len(hiddenNums)

        # Create and initialize delta
        self.delta = [[]] * (len(hiddenNums) + 2)
        self.delta[0] = [0.0] * inputNum
        for rowNum in range(1, len(hiddenNums) + 1):
            self.delta[rowNum] = [0.0] * hiddenNums[rowNum - 1]
        self.delta[len(hiddenNums) + 1] = [0.0] * outputNum

        # Create and initialize activation
        self.activation = [[]] * (len(hiddenNums) + 2)
        self.activation[0] = [0.0] * inputNum
        for rowNum in range(1, len(hiddenNums) + 1):
            self.activation[rowNum] = [0.0] * hiddenNums[rowNum - 1]
        self.activation[len(hiddenNums) + 1] = [0.0] * outputNum


        # Initialize inputlayer
        for num in range(0, inputNum):
            newInputNode = InputNode.InputNode(0.0)
            self.inputLayer.append(newInputNode)

        # Initialize hidden layers
        for num in range(0, len(hiddenNums)):
            # If the length of the current hidden layer is 0, initialize the layer to be full of null variables
            if (len(self.hiddenLayer[num]) == 0):
                self.hiddenLayer[num] = [None] * hiddenNums[num]

            # Holds the previous layer in the neural network
            previousLayer = []

            # If this is the first hidden layer, the preceding layer is the input layer
            if (num == 0):
                for inputNum in range(0, len(self.inputLayer)):
                    previousLayer.append(self.inputLayer[inputNum])
            # Otherwise, it's the preceding hidden layer
            else:
                for hiddenNum in range(0, len(self.hiddenLayer[num - 1])):
                    previousLayer.append(self.hiddenLayer[num - 1][hiddenNum])

            # For every node in the current hidden layer 
            for currNum in range(0, hiddenNums[num]):
                
                # Create a dummy node with a random weight and an activation that is always 1 and connect it to the hidden node
                hiddenNode = HiddenNode.HiddenNode(False)

                dummyNode = InputNode.InputNode(1.0)
                dummyWeight = rand.gauss(0, 1)
                dummyConnection = Connection.Connection(
                    dummyNode, hiddenNode, dummyWeight)
                hiddenNode.addIncomingConnection(dummyConnection)
                dummyNode.addOutgoingConnection(dummyConnection)

                # Make a connection from every node in the previous layer to this node
                for prevNum in range(0, len(previousLayer)):
                    weight = rand.gauss(0, 1)
                    connection = Connection.Connection(
                        previousLayer[prevNum], hiddenNode, weight)
                    previousLayer[prevNum].addOutgoingConnection(connection)
                    hiddenNode.addIncomingConnection(connection)

                # Add this node to the current hidden layer
                self.hiddenLayer[num][currNum] = (hiddenNode)

        # Initialize the output layer to null values
        self.outLayer = [None] * outputNum
        # Iterate a number of times equal to the amount of nodes that should be in the output layer
        for num in range(0, outputNum):
            # Create a new output node
            newOutputNode = OutputNode.OutputNode(True)

            # Create a new dummy node with random weight and activation that is always 1 and connect it to this node
            dummyNode = InputNode.InputNode(1.0)
            dummyWeight = rand.gauss(0, 0.1)
            dummyConnection = Connection.Connection(
                dummyNode, newOutputNode, dummyWeight)

            newOutputNode.addIncomingConnection(dummyConnection)
            dummyNode.addOutgoingConnection(dummyConnection)

            # The layer before the output layer is always the last hidden layer
            previousLayer = self.hiddenLayer[len(self.hiddenLayer) - 1]
            
            # Iterate through all the nodes in the last hidden layer and add connections from them to the current output node
            for prevNum in range(0, len(previousLayer)):
                weight = rand.gauss(0, 0.1)
                connection = Connection.Connection(
                    previousLayer[prevNum], newOutputNode, weight)
                previousLayer[prevNum].addOutgoingConnection(connection)
                newOutputNode.addIncomingConnection(connection)

            # Add the current output Node to the output layer
            self.outLayer[num] = (newOutputNode)

    # Function that allows a Neural Network to print itself
    def printNN(self):
        # Skip input nodes (they aren't that interesting)
        # Iterate through hidden nodes
        for num in range(0, len(self.hiddenLayer)):
            for hiddenNum in range(0, len(self.hiddenLayer[num])):
                # Print the hidden unit and all its incoming connections, with their weights
                print("Hidden Unit: %d %d \n" % (num, hiddenNum))
                if (self.hiddenLayer[num][hiddenNum].getIncomingConnections() != None):
                    for conNum in range(0, len(self.hiddenLayer[num][hiddenNum].getIncomingConnections())):
                        print("Connection: %d to %d : %s \n" % (
                            conNum, hiddenNum, self.hiddenLayer[num][hiddenNum].getIncomingConnections()[conNum].getWeight()))
        # Iterate through output nodes
        for outNum in range(0, len(self.outLayer)):
            # Print the output node and all its incoming connections, with their weights
            print("Output Unit: %d\n" % outNum)
            if(self.outLayer[outNum].getIncomingConnections() != None):
                for conNum in range(0, len(self.outLayer[outNum].getIncomingConnections())):
                    print("Connection: %d to %d : %f\n" % (
                        conNum, outNum, self.outLayer[outNum].getIncomingConnections()[conNum].getWeight()))

    
    # Finds the softmax values for the output layer in a neural network
    def findSoftmaxActivation(self, trace):
        # Holds the incoming sums of every output node
        incomingSums = []

        # Iterate through output nodes
        for output in range(0, len(self.outLayer)):
            # Get the sum of the incoming connections for each node and append it to incoming sum
            activationForNode = self.outLayer[output].findIncomingSum()
            incomingSums.append(activationForNode)

        # Subtract the maximum value of incomingSums from every value in incomingSums 
        # This ensures we don't get any NaN values when we take the exponent
        maxVal = max(incomingSums)
        for a in range(0, len(incomingSums)):
            incomingSums[a] = incomingSums[a] - maxVal

        # Holds the activations, first non-softmax and then softmax for each output node
        softMaxActivations = []

        # Iterate through the nodes in the output layer
        for a in range(0, len(self.outLayer)):
            # Print informative message
            if (trace):
                print("%d: %f" % (a, np.exp(incomingSums[a])))
            # Append e to the power of the (altered) incoming sum of this node and append it to Softmax Activations
            softMaxActivations.append(np.exp(incomingSums[a]))

        # Holds the sum of the non-softmax activations for the output layer
        sumVal = sum(softMaxActivations)
        # Print informative message
        if (trace):
            print("sum: %f" % sumVal)

        # Iterate through the output layer
        for a in range(0, len(incomingSums)):
            # Divide each activation by the sum
            softMaxActivations[a] = softMaxActivations[a] / sumVal
            # print informative message
            if (trace):
                print("y%d: %f" % (a, softMaxActivations[a]))

        # iterate through the output layer
        for output in range(0, len(self.outLayer)):
            # set the activation of each output node to be the softmax activation for that node
            activationForNode = softMaxActivations[output]
            self.outLayer[output].setActivation(activationForNode)
            # Print informative message
            if (trace):
                print("activation: %f" % activationForNode)

        return softMaxActivations

    # Train Neural Network using Back-Propagation Learning, for given batch of examples, learning rate, instruction whether or not to print an informative message 
    def trainBatch(self, data, learningRate, trace):

        # iterate through the batch 
        for datumItr in range(0, len(data)):

            # Print informative message
            if (trace):
                print("Example: \n")
                data[datumItr].printDatum()

            # Holds an individual data point in the batch
            datum = data[datumItr]

            # If the number of nodes in the input layer is not equal to the number of parameters in the data point, something has gone very wrong
            if (len(self.inputLayer) != datum.getParameterCount()):
                if (trace):
                    print(
                        "Error: Number of input nodes should be = to number of parameters.\n")
                    return None

            # Iterate through input layer, set activation within network and outside database to example value
            for inputItr in range(0, len(self.inputLayer)):
                self.inputLayer[inputItr].setActivation(
                    datum.getParameter(inputItr))
                self.activation[0][inputItr] = datum.getParameter(inputItr)
                # Print informative message..
                if (trace):
                    print("Input %d initialized to %f...\n" %
                          (inputItr, datum.getParameter(inputItr)))

            # Iterate through hidden layers
            for layerItr in range(0, len(self.hiddenLayer)):
                # Iterate through nodes in hidden layers
                for hiddenItr in range(0, len(self.hiddenLayer[layerItr])):
                    # Propagate activation values through the network, store in outside database
                    activationForNode = self.hiddenLayer[layerItr][hiddenItr].findActivation(
                    )
                    self.activation[layerItr +
                                    1][hiddenItr] = activationForNode
                    # Print informative message
                    if (trace):
                        print("Hidden Node, Layer: %d, Node %d, Value: %f" % (
                            layerItr, hiddenItr, activationForNode))

            # Retrieve the softmax activations for each output node
            softMaxActivations = self.findSoftmaxActivation(False)

            # Store the softmax activations for each output node in the activations list corresponding to the output layer
            self.activation[len(self.activation) - 1] = softMaxActivations

            # Stores the of the list representing the output layer in activation
            final = len(self.activation) - 1

            # Print informative message
            if (trace):
                for aItr in range(0, len(self.outLayer)):
                    print("Output Node %d, Value %f" %
                          (aItr, self.activation[final][aItr]))

            # Iterate through the nodes in the output layer
            for outputItr in range(0, len(self.outLayer)):
                # Stores gradient value for node in output layer
                deltaNum = (datum.getType(outputItr) -
                            self.activation[final][outputItr])

                # Print informative message
                if (trace):
                    print("t: %f; activation: %f; error (output): %f\n" % (
                        datum.getType(outputItr), self.activation[final][outputItr], deltaNum))

                # store gradient value
                self.delta[final][outputItr] = (deltaNum)

            # Propagate gradients back through hidden layers
            for layerItr in range((len(self.hiddenLayer) - 1), -1, -1):
                # Print informative message
                if (trace):
                    print("Hidden Layer %d\n" % layerItr)
                # For node in current hidden layer
                for hiddenItr in range(0, len(self.hiddenLayer[layerItr])):
                    # Get derivative of tanh
                    der = self.hiddenLayer[layerItr][hiddenItr].derivative()

                    # Stores the sum of the gradient * weight values of all the outgoing connections
                    sumTotal = 0.0

                    # Iterate through outgoign connections
                    for outCon in range(0, len(self.hiddenLayer[layerItr][hiddenItr].getOutgoingConnections())):
                        con = self.hiddenLayer[layerItr][hiddenItr].getOutgoingConnections()[
                            outCon]

                        # Print informative message
                        if (trace):
                            print("%f * %f = %f" % (self.delta[layerItr + 2][outCon], con.getWeight(
                            ), (self.delta[layerItr + 2][outCon] * con.getWeight())))

                        # Adds gradient * weight of outgoing connection to the sum total
                        sumTotal += self.delta[layerItr +
                                               2][outCon] * con.getWeight()
                    # Print informative message
                    if (trace):
                        print("der: %f; sum: %f; error (hidden): %f\n" %
                              (der, sumTotal, (der * sumTotal)))

                    # Add gradient to gradient database
                    self.delta[layerItr + 1][hiddenItr] = der * sumTotal

            # Iterate through number of layers in network, starting at first hidden layer
            for layerItr in range(1, len(self.delta)):
                # Print informative message
                if (trace):
                    print("Layer %d\n" % layerItr)

                # Nodes in current layer
                currentLayer = []

                # If output layer, initialize current layer with output nodes
                if (layerItr == len(self.delta) - 1):
                    for outputItr in range(0, len(self.outLayer)):
                        currentLayer.append(self.outLayer[outputItr])
                # Otherwise, initialize current layer with nodes from current hidden layer
                else:
                    for hiddenItr in range(0, len(self.hiddenLayer[layerItr - 1])):
                        currentLayer.append(
                            self.hiddenLayer[layerItr - 1][hiddenItr])

                # For nodes in the current layer
                for currItr in range(0, len(currentLayer)):
                    # Print informative message
                    if (trace):
                        print("Node %d (Delta: %f)\n" %
                              (currItr, self.delta[layerItr][currItr]))

                    # Get current node
                    outGoingNode = currentLayer[currItr]

                    # Print informative message
                    if (trace):
                        outGoingNode.printNode()

                    # Iterate through incoming connections
                    for incItr in range(0, len(outGoingNode.getIncomingConnections())):
                        # Retrieve current connection
                        incCon = outGoingNode.getIncomingConnections()[
                            incItr]

                        # Retrieve node at incoming end of connection
                        incomingNode = incCon.getInNode()

                        # Initialize activation of node to 0
                        activationForNode = 0.0

                        # If dummy node, get activation from node itself
                        if (incItr == 0):
                            activationForNode = incomingNode.getActivation()
                        # Otherwise, get activation from outside database
                        else:
                            activationForNode = self.activation[layerItr - 1][incItr - 1]
                        # Update weight of current node:
                        # w = w + learningRate * gradient (w/Cross Entropy + SoftMax)
                        updatedWeight = incCon.getWeight() + (learningRate * activationForNode * self.delta[layerItr][currItr])

                        # Print informative message
                        if (trace):
                            print("Update Connection %d to %d, Current weight: %f + %f * %f * %f to %f" % (currItr, incItr, incCon.getWeight(
                            ), learningRate, activationForNode, self.delta[layerItr][currItr], updatedWeight))

                        # Set weight of connection to updated number
                        incCon.setWeight(updatedWeight)

    # Train a Neural Network using Backpropagation Learning on a given set of batches (one batch at at a time) until told to stop with a given learning rate, then test on a given training set and given development set
    def train(self, batches, exampleList, testList, stop,learningRate, trace):
        # Stores current iteration
        itr = 0

        # Stores errors and f1 scores for each iteration
        errors = []
        
        # While iteration is less than target
        while itr < stop:
            # Retrieves a random batch index
            batchNum = rand.randrange(0, len(batches), 1)
            # Print informative message
            if (trace):
                print(
                    "-----------------------------Iteration: %d-----------------------------\n" % itr)
            # Print iteration number
            print("%d" %itr)

            # Retrieves the batch using the random batch index
            currBatch = batches[batchNum]

            # Train the Neural Network on the random batch, using the given learning Rate
            self.trainBatch(currBatch, learningRate, trace)
            # Increase the iteration number by 1
            itr += 1

            # Initialize the f1 score to 0
            f1 = 0
            # Attempt to get the f1 score for the training set
            # (because of divide by 0 errors, this may not work)
            # However, that usually means the NN is currently really bad, so I just return 0
            try:
                f1 = self.f1score(exampleList, False)
            except ZeroDivisionError:
                pass

            # Calculate the f1 score for the development set
            f12 = 0
            try:
                f12 = self.f1score(testList, False)
            except ZeroDivisionError:
                pass

            # Append both f1 scores to the errors list
            errors.append(f1)
            errors.append(f12)

            # Get the accuracy for the neural network on the training set and development set
            acc1 = self.testData(exampleList, False, True)
            acc2 = self.testData(testList, False, True)

            # Append the accuracy values to the errors list
            errors.append(acc1) 
            errors.append(acc2)

        # return the errors list
        return errors
            

    # Test the Accuracy on the Neural Network on a Single Example 
    def testDatum(self, datum, trace):
        # Gets the total loss for the example (using L1)
        # L1 = | actual output - Neural Network value |
        totalLossForExample = 0.0

        # Iterate through input layer, set activation within network to example value
        for param in range(0, datum.getParameterCount()):
            self.inputLayer[param].setActivation(datum.getParameter(param))

        # Forward propagate activation through hidden layers
        for layer in range(0, len(self.hiddenLayer)):
            for hiddenNode in range(0, len(self.hiddenLayer[layer])):
                self.hiddenLayer[layer][hiddenNode].findActivation()

        # Store the softmax Activations for the final layer
        softMaxActivations = self.findSoftmaxActivation(trace)

        # For every output in the final layer
        for output in range(0, len(self.outLayer)):
            # Get the corresponding "correct" value from the example
            y = datum.getType(output)

            # Retrieve the activation for the output node
            activationForNode = softMaxActivations[output]
            # Print informative message
            if (trace):
                print("Input: ")
                for param in datum.getParameters():
                    print("%f " % param)
                print("NN Output: %f\nActual Answer: %f\nAmount of Error: %f\n" %(activationForNode, y, abs((y - activationForNode))))
            # Add L1 loss to total loss for example
            totalLossForExample += abs((y - activationForNode))

        return totalLossForExample

    # Test the Accuracy of the NN using L1 loss on a given set of examples
    def testData(self, dataSet, trace, percentage):
        # Total loss initialized to 0
        totalLoss = 0.0

        # Iterate through examples in dataset
        for ex in range(0, len(dataSet)):
            # Add loss for single example to total loss
            totalLoss += self.testDatum(dataSet[ex], trace)

        # If we want the accuracy of the NN as a percentage of the dataset, we subtract the average loss from 1
        if (percentage):
            totalLoss = 1 -(totalLoss / (len(dataSet)))

        # We return the total loss (in whichever form it takes)
        return totalLoss

    # Retrieve whether the classification for a single data point is a false positive, false negative, true positive, or true negative
    def f1Datum(self, datum, trueForPosNeg, trace):
        # Iterate through input layer, set activation within network to example value
        for param in range(0, datum.getParameterCount()):
            self.inputLayer[param].setActivation(datum.getParameter(param))

        # Forward propagate activation through hidden layers
        for layer in range(0, len(self.hiddenLayer)):
            for hiddenNode in range(0, len(self.hiddenLayer[layer])):
                self.hiddenLayer[layer][hiddenNode].findActivation()

        # Get the softmax activations for the output layer
        softMaxActivations = self.findSoftmaxActivation(False)
        # print informative message
        if (trace):
            print(datum.__str__())
            print("Activation: %f" %softMaxActivations[0])

        # Get the binary 'type' of the example (i.e., whether it is positive or negative)
        binaryy = datum.getType(0)
        if (binaryy == 1):  # Data is positive
            # Activation is positive (true positive)
            if (softMaxActivations[0] == max(softMaxActivations)):
                # Print informative message
                if (trace):
                    print("True Positive!")
                # Add a tally to the "true positive" index of the trueForPosNeg tuple
                trueForPosNeg[0] += 1
            else:  # Activation is negative (false negative)
                # Print informative message
                if (trace):
                    print("False Negative!")
                # Add a tally to the "false negative" index of the trueForPosNeg tuple
                trueForPosNeg[3] += 1

        elif (binaryy == 0):  # Data is negative
            # Activation is negative (true negative)
            if (softMaxActivations[0] != max(softMaxActivations)):
                # Print informative message
                if (trace):
                    print("True Negative!")
                # Add a tally to the "true negative" index of the trueForPosNeg tupe
                trueForPosNeg[1] += 1
            else:  # Activation is positive (false positive)
                # Print informative message
                if (trace):
                    print("False Positive!")
                # Add a tally to the "false positive" index of the trueForPosNeg tuple
                trueForPosNeg[2] += 1


        if (trace):
            print("\n")

        # return trueForPosNeg tuple
        return trueForPosNeg

    # Calculate the f1 Score for a dataset
    def f1score(self, dataSet, trace):
        # F1 = 2PR/(P + R) where
        # P = # of true positives / (# true positives + false positives)
        # R = # of true positives / (# true positives + false negatives)

        # Create a tuple to hold:
        # 1. The number of true positives
        # 2. The number of false positives
        # 3. The number of true negatives
        # 4. The number of false negatives
        truePosNeg = [0, 0, 0, 0]

        # Iterate through the dataset
        for ex in range(0, len(dataSet)):
            # Calculate whether the classification for the dataset was a false/true pos/neg
            truePosNeg = self.f1Datum(dataSet[ex], truePosNeg, trace)

        # Print informative message
        if (trace):
            print("True Positives: %d\nFalse Positives: %d\nTrue Negatives: %d\nFalse Negatives: %d\nLength of dataset: %d" %(truePosNeg[0], truePosNeg[2], truePosNeg[1], truePosNeg[3], len(dataSet)))
        
        # Calculate p
        p = truePosNeg[0] / (truePosNeg[0] + (truePosNeg[2]))
        # Print informative message
        if (trace):
            print("P: %f" % p)

        # Calculate r
        r = truePosNeg[0] / (truePosNeg[0] + (truePosNeg[3]))
        # Print informative message
        if (trace):
            print("R: %f\n" % r)

        # Calculate f1
        f1 = (2 * p * r) / (p + r)
        # Print informative message
        if (trace):
            print("F1 = %f" % f1)

        # Return f1
        return f1

    # DEFUNCT -- Function to test accuracy and f1 score simultaneously
    def testAccAndF1ScoreSimultaneously(self, dataSet, trace):
        totalLoss = 0.0

        truePosNeg = [0, 0, 0, 0]

        for ex in range(0, len(dataSet)):
            truePosNeg = self.f1Datum(dataSet[ex], truePosNeg, trace)
            totalLoss += self.testDatum(dataSet[ex], trace)

        totalLoss = (1 - (totalLoss / (len(dataSet) * 2)))


        if (trace):
            print("True Positives: %d\nFalse Positives: %d\nTrue Negatives: %d\nFalse Negatives: %d\nLength of dataset: %d" %(truePosNeg[0], truePosNeg[2], truePosNeg[1], truePosNeg[3], len(dataSet)))
        p = truePosNeg[0] / (truePosNeg[0] + (truePosNeg[2]))
        if (trace):
            print("P: %f" % p)

        r = truePosNeg[0] / (truePosNeg[0] + (truePosNeg[3]))
        if (trace):
            print("R: %f\n" % r)

        f1 = (2 * p * r) / (p + r)
        if (trace):
            print("F1 = %f" % f1)
        
        answers = [f1, totalLoss]

        return answers

# Utility function to print an Array
def printArrayPretty(array):
    print("Array: ")
    # Iterate through indices of array, print string representation of whatever is at index
    for itr in range(0, len(array)):
        print("Index: %d\n%s " % (itr,array[itr].__str__()))
    print("\n")

# Utility Function to Print 2D Array
def printArray2Dpretty(array):
    print("2D Array: ")
    # Iterate through indices of array, print array at index
    for itr in range(0, len(array)):
        printArrayPretty(array[itr])

# Trains a Neural Network using backprop on the given set of batches using the given learning rate for the given number of epochs
# For each epoch, graphs the accuracy (calculated using L1), and F1 score for the test and training sets
# Finally, prints the trained neural network to a specified output file
def trainAndPlot(neuralnetwork, exampleList, testList, batchSet, stop, learningRate, outputFile):
 
    # Gets the F1 and accuracy scores for each epoch, while training NN
    errorForEpochs = neuralnetwork.train(batchSet, exampleList, testList, stop, learningRate, False)

    # Label x axis of graph 
    plt.xlabel("Number of Epochs (Batch Size = 10)")
    
    # Label y axis of graph
    plt.ylabel("Error (as f1 score)")
     
    # Label graph title
    plt.title("Change in Error over Number of Epochs")

    # Plot F1 scores for test and training sets over each epoch
    plt.plot(range(0, stop), errorForEpochs[::4], color = 'red')
    plt.plot(range(0, stop), errorForEpochs[1::4], color = 'blue')
    plt.legend(['Training Data', 'Development Data'])

    plt.show()

    # Label y axis of graph
    plt.ylabel("Error (as fraction of DataSet)")

    # Plot accuracy scores for test and trainig sets over each epoch
    plt.plot(range(0, stop), errorForEpochs[2::4], color = 'orange')
    plt.plot(range(0, stop), errorForEpochs[3::4], color = 'green')
    plt.legend(['Training Data', 'Development Data'])

    plt.show()

    # Print Neural Network to file
    printNNtoFile(neuralnetwork, outputFile)

# Read Neural Network in from given file
def readInNN(inputFile):

    # Open file 
    input = open((os.path.join(sys.path[0], inputFile)), "r")
    # Get lines of file as a list
    inputLines = list(input)

    # The count of the input layer is written on the first line
    inputLayerCnt = int(inputLines[0].replace("\s+", ""))
    # The count of each hidden layer is written on the second line
    hiddenLayers = inputLines[1].split("\s+")
    hiddenLayersCnt = []
    for layer in hiddenLayers:
        hiddenLayersCnt.append(int(layer))

    # The count of the output layer is written on the third line
    outputLayerCnt = int(inputLines[2].replace("/s+", ""))

    # Create and initialize a neural layer from all of these counts
    nn = NeuralNetwork(inputLayerCnt, outputLayerCnt, hiddenLayersCnt)

    # Set all of the weight values for each incoming connection of each hidden node in each hidden layer
    currentLine = 3
    for layer in range(0, len(hiddenLayersCnt)):
        for node in range(0, hiddenLayersCnt[layer]):
            currentLine += 1
            incomingConnectionStr = inputLines[3 + layer + node]
            incomingConnectionWeights = incomingConnectionStr.split()
            # print(len(incomingConnectionWeights))
            # printArrayPretty(incomingConnectionWeights)
            incomingConnections = nn.hiddenLayer[layer][node].getIncomingConnections()
            # print(incomingConnectionWeights[2])
            # print("Length: %d" %(len(incomingConnectionWeights)))
            for incCon in range(0, len(incomingConnections)):
                # print("Number: %d" %incCon)
                incomingConnections[incCon].setWeight(float(incomingConnectionWeights[incCon]))
    
    # Set all of the weight values for each incoming connection of each output node in the output layer
    for node in range(0, outputLayerCnt):
        incomingConnectionStr = inputLines[currentLine + node]
        incomingConnectionWeights = incomingConnectionStr.split()
        # printArrayPretty(incomingConnectionWeights)
        incomingConnections = nn.outLayer[node].getIncomingConnections()
        # printArrayPretty(incomingConnections)
        for incCon in range(0, len(incomingConnections)):
            incomingConnections[incCon].setWeight(float(incomingConnectionWeights[incCon]))

    # Return the neural network
    return nn

# Print a given neural network to a given output file
def printNNtoFile(nn, outputFile):
    # open given file to write to it
    output = open(outputFile, "w")

    # Print the number of nodes in the input layer
    inputLayerCnt = "{0}\n".format(len(nn.inputLayer))
    output.write(inputLayerCnt)

    # Print the number of nodes in each hidden layer on the second line
    for layer in nn.hiddenLayer:
        currLayerCnt = "{0} ".format(len(layer))
        output.write(currLayerCnt)

    output.write("\n")

    # Print the number of output nodes on the third line
    outputLayerCnt = "{0}\n".format(len(nn.outLayer))
    output.write(outputLayerCnt)

    # Iterate through all the nodes in the hidden layers, and print the weights of each incoming connection
    for layer in range(0, len(nn.hiddenLayer)):
        for node in range(0, len(nn.hiddenLayer[layer])):
            incomingConnections = nn.hiddenLayer[layer][node].getIncomingConnections()
            for connection in incomingConnections:
                currentWeight = "{0} ".format(connection.getWeight())
                output.write(currentWeight)
            output.write("\n")
    
    # Iterate through all the nodes in the output layer and print the weights of each incoming connection
    for node in range(0, len(nn.outLayer)):
        incomingConnections = nn.outLayer[node].getIncomingConnections()
        for connection in incomingConnections:
            currentWeight = "{0} ".format(connection.getWeight())
            output.write(currentWeight)
        output.write("\n")

    # Close the file
    output.close()
    
# DEFUNCT -- Tests best linearsmoke model
def testBestModel():
    nn = readInNN("BestNNModel.txt")

    exampleSet = DataReader.DataReader("dataProject2/data/linearSmoke")
    # printArrayPretty(exampleSet.exampleList)

    accuracy = (1 - nn.testData(exampleSet.exampleList, True, True)) * 100

    # f1 = nn.f1score(exampleSet.exampleList, True)

    print("Accuracy (as Percentage of Dataset): %f" %accuracy)
    # print("F1 Score: %f" %f1)

# Function to run inference engine for given input file
def inferenceEngine():
    
    # Print Welcome Message
    print("Welcome to the Inference Engine!")
    userInput = ""
    userOutput = ""
    nn = None
    # Prompt user for filepath to retrieve NN to test inference engine on
    print("Input a filepath to read the Neural Network from: ")
    nnFile = input()
    if (nnFile.upper() == "QUIT"):
        userInput = "QUIT"
    else:
        try:
            nn = readInNN(nnFile)
        except:
            print("Error: Could not read in Neural network from file.")
            print("Will initialize from current best Neural Network Model.")
            nn = readInNN("NNModel")

    # If user has not already input quit
    while(userInput.upper() != "QUIT"):
        # Prompt user for example parameters
        print("Please enter the input to an example you wish to test.")
        print("Preferably, do this by writing the input values on a single line with a single space between each value. Ex:")
        print("1 2 3 4")
        print("Additionally, remember you can enter 'QUIT' to quit at any time.")
        # Read in user parameters
        userInput = input("Input: ")
        # print(userInput.upper())
        # Check if user entered quit
        if (userInput.upper() == "QUIT"):
            break
        # Prompt user for type
        print("Now enter the output as 0 or 1. Ex:")
        print("1")
        print("Additionally, remember you can enter 'QUIT' to quit at any time.")
        # Read in user input
        userOutput = input("Output: ")
        # Check if input is quit
        if (userOutput.upper() == "QUIT"):
            break
        
        # Type check input values
        inputStrs = userInput.split()
        if(len(inputStrs) != len(nn.inputLayer)):
            print("Error: Please enter the correct number of inputs (%d)!" %len(nn.inputLayer))
            continue

        inputVals = []
        for str in inputStrs:
            try:
                inputVals.append(float(str))
            except ValueError:
                print("Input %s was not convertable into a float." %str)
        if(len(inputVals) < len(inputStrs)):
            continue

        # Type check output values
        type = -1.0
        try:
            type = float(userOutput)
        except ValueError:
            print("Output %s was not convertable into a float." %userOutput)
            continue

        # Create example with input parameters and type
        datum = Datum.Datum()
        datum.setParameters(inputVals)
        datum.setType(type)

        # Test it using L1 Loss
        print("TESTING...\n")

        nn.testDatum(datum, True)

    print("Goodbye!")

# Function to train a neural net using specifications input from the terminal
def trainFromTerminal():
    print("Welcome to the Training Program!")
    userInput = ""
    while (userInput.upper() != "QUIT"):
        print("Choose your method of Neural Network initialization:")
        print("1. Read in from file (input '1')")
        print("2. Start with a brand new Neural Network (input '2')")
        print("Or enter 'quit' to quit the Training Program.")
        
        userInput = input()
        nn = None
        
        if (userInput.upper() == "QUIT"):
            break
        
        userChoice = -1
        try:
            userChoice = int(userInput)
        except:
            print("Error: %s was not an integer value." %userInput)
            continue
        
        if (userChoice == 1):
            print("Input a filepath to read the Neural Network from: ")
            nnFile = input()
            if (nnFile.upper() == "QUIT"):
                break
            try:
                nn = readInNN(nnFile)
            except:
                print("Error: Could not read in Neural network from file.")
                break
        elif (userChoice == 2):
            print("Input an integer number of Hidden Layers that you desire your network to have:")
            layerStr = input()
            layerNum = 0


            if (layerStr.upper() == "QUIT"):
                break
            try:
                layerNum = int(layerStr)
            except:
                print("Layer number %s was not an integer.")
                break
            
            hiddenNums = []
            for layer in range(0, layerNum):
                print("Input the integer number of hidden nodes you wish to have in this layer: ")
                currNodeInput = input()
                currNodeNum = 0

                if (currNodeInput.upper() == "QUIT"): 
                    break
                try:
                    currNodeNum = int(currNodeInput)
                    hiddenNums.append(currNodeNum)
                except:
                    print("Node number was not an integer.")
                    break
            # print("Hidden num length: %d; layer num: %d" %(len(hiddenNums), layerNum))
            if (len(hiddenNums) != layerNum):
                break
            else:
                nn = NeuralNetwork(8, 2, hiddenNums)
        else:
            print("Error: %d was not a valid choice ('1' or '2')")

        if (nn == None):
            break
        
        learningRate = 0
        print("Now, input the power of e that you wish the learning rate to be (ex. -1):")

        learningStr = input()

        if (learningStr.upper() == "QUIT"):
            break
        try:
            learningRate = int(learningStr)
        except:
            print("Error: %s was not an integer." %learningStr)
            break
        
        batchSize = 0
        print("Now input the integer number that you wish the size of the batches to be: (ex. 10)")

        batchStr = input()
        
        if (batchStr.upper() == "QUIT"):
            break

        try:
            batchSize = int(batchStr)
        except:
            print("Error: %s was not an integer." %batchStr)
            break

        epochs = 0
        print("Finally, input the integer number of epochs that you wish the Neural Network to train for: (ex. 500)")

        epochStr = input()

        if (epochStr.upper() == "QUIT"):
            break
        try:
            epochs = int(epochStr)
        except:
            print("Error: %s was not an integer." %epochStr)

        outputFile = ""
        print("Input a filepath to read the Neural Network out to: (ex. 'outputFile')")
        outputFile = input()

        try:
            open(outputFile, "w")
        except:
            print("Error: Invalid Filepath.")
            break

        htru2train = DataReader.DataReader("dataProject2/data/htru2.train")

        htru2dev = DataReader.DataReader("dataProject2/data/htru2.dev")

        htru2trainbatches = htru2train.partitionIntoBatches(batchSize)

        trainAndPlot(nn, htru2train.exampleList, htru2dev.exampleList, htru2trainbatches, epochs, learningRate, outputFile)

# Terminal function that allows access to both training program and inference engine
def terminal():
    print("Welcome to the Terminal!")
    userInput = ""
    while(userInput.upper() != "QUIT"):
        print("Please pick a functionality:")
        print("1. Train a new Neural Network (input '1')")
        print("2. Run the Inference Engine (input: '2')")
        print("Or enter 'quit' to quit the Terminal.")

        userInput = input()
        
        if (userInput.upper() == "QUIT"):
            break
        
        userChoice = -1
        try:
            userChoice = int(userInput)
        except:
            print("Error: %s was not an integer value." %userInput)
            continue
        
        if (userChoice == 1):
            trainFromTerminal()
        elif (userChoice == 2):
            inferenceEngine()
        else:
            print("Error: %d was not a valid choice ('1' or '2')")
        
# dataReader = DataReader.DataReader("dataProject2/data/htru2.train")

# dataReader2 = DataReader.DataReader("dataProject2/data/htru2.dev")

# batches = dataReader.partitionIntoBatches(len(dataReader.exampleList) / 2)
# printArrayPretty(batches[0])
# printArrayPretty(batches[2])

# nn = NeuralNetwork(8, 2, [100] * 1)
# nn.train(batches, dataReader.exampleList, 1000, np.exp(-2), False)

# nn.printNN()

# printNNtoFile(nn, "BestNNModel.txt")
# nn = readInNN("newNNModel1")

# newNN.printNN()

# inferenceEngine()

# dataList = []
# dataList.append(dataReader.exampleList[0])

# trainAndPlot(nn, dataReader.exampleList, dataReader2.exampleList, batches, 50, np.exp(-3), "newNNModel1.1")

# nn.train(batches, 100, np.exp(-1), False)

# dataReader.printData()

# totalLoss = (1 - nn.testData(dataReader.exampleList, False, True)) * 100

# f1 = nn.f1score(dataReader2.exampleList, False)

# print("F1 Score: %f\n" %f1)

# nn.printNN()

# print("Total Loss (in Percentage): %f%%\n" % (totalLoss))

# nn.printNN()

terminal()
