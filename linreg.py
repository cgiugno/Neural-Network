import math
import numpy as np
import matplotlib.pyplot as plt

# Method that takes input X values and powers and returns the Design Matrix
def makePhi(xV, m):
    # Initializes the Design Matrix to an array that will later be turned into a matrix
    xBasisList = np.ndarray(shape=(len(xV), m + 1), dtype= float)
    # For every data point in the input X values
    for i in range(0, len(xV)):
        # Iterate from 0 to power
        for j in range(0, m + 1):
            # The Basis Value at the ijth index of Design Matrix is the ith x value to the power j
            xBasisList[i][j] = xV[i] ** j
    # Make the 'list' version of the Design Matrix into the 'matrix' version of the Design Matrix
    xBasis = np.matrix(xBasisList) 
    # print(xBasis)
    return xBasis

# Method that creates a t value matrix with the proper shape
def makeTValueMatrix(yV):
    yMatrix = np.matrix(yV).T
    return yMatrix

# Method that generates the a matrix of the maximum likelihood weights given an input:
# xV: x Values
# yV: t Values (apologies--I'm so used to the x/y notation that I confused the two)
# m: the power
# lr: the regularization constant (lambda)
def makeWML(xV, yV, m, lr):

    # Creates the Design Matrix
    phi = makePhi(xV, m)

    # Creates the phi.T phi term
    wML1inner = np.matmul(phi.T, phi)
 
    # Creates the term with the identity and regularization constant
    wMLidentity = (lr * np.identity(m + 1))

    # Creates the first term of the Maximum Likelihood Weights equation
    wML1 = np.linalg.inv(np.add(wMLidentity,  wML1inner))
    
    # Creates the second term of the Maximum Likelihood equation
    wML2 = np.matmul(phi.T, makeTValueMatrix(yV))
    
    # Creates the Weights Matrix from the multiplication of the first and second term of the Maximum Likeilhood Weights equation
    wML = np.matmul(wML1, wML2)
    
    return wML

# DEFUNCT -- LIKELIHOOD METHOD
def likelihood(xV, yV, weights, power, beta):
    print("First term", (len(xV) / 2) * math.log(beta))
    print("Second Term", (- ((len(xV) / 2) * math.log(2 * math.pi))))
    print("Third Term", (- (beta) * sumOfSquaresError(xV, yV, power, weights)))
    totalLikelihood = ((len(xV) / 2) * math.log(beta)) - ((len(xV) / 2) * math.log(2 * math.pi)) - ((beta) * sumOfSquaresError(xV, yV, power, weights))
    
    return totalLikelihood

# DEFUNCT -- BETA METHOD
def beta(xV, yV, power, mL):
    totalBeta = 0
    for i in range(0, len(xV)):
        wphi = 0
        for j in range(0, power + 1):
            wphi += (mL[j][0].item(0) * (xV[i] ** j)) 
        totalBeta += (yV[i] - wphi) ** 2
        print("Beta Function %f %f %f %f %f" %(xV[i], wphi, yV[i], ((yV[i] - wphi) ** 2), totalBeta))
    totalBeta = totalBeta / len(xV)
    print(len(xV))
    return 1/ totalBeta

# Method that generates the mean sum-of-squares error for a given:
# xV: input values
# yV: target values
# power: power, of course
# mL: maximum likelihood weights
def sumOfSquaresError(xV, yV, power, mL):
    # Initializes total error to 0
    totalSoSError = 0
    # Iterates through data points
    for i in range(0, len(xV)):
        # Value calculated from the maximum likelihood and current input
        wphi = 0
        # Multiply x^0...power by the weights
        for j in range(0, power + 1):
            wphi +=  (mL[j][0].item(0) * (xV[i] ** j))
        # Subtract the result from the target value and square it
        totalSoSError += (yV[i] - wphi) ** 2
    # Divide the total sum-of-squares error by the number of data points to get the *mean* sum-of-squares error
    totalSoSError = (totalSoSError / len(xV))
    return totalSoSError

# Method that generates the mean sum-of-squares error for a given:
# xV: input values
# yV: target values
# power: power, of course
# mL: maximum likelihood weights
# lr: regularization constant (lambda)
def sumOfSquaresErrorWithRegularization(xV, yV, power, mL, lr):
    # Initializes total error to 0
    totalSoSError = 0
    # Initializes the total regularization error to 0
    magnituteMLSquared = 0
    # Iterates from 0 to power
    for j in range(0, power + 1):
        # Adds the absolute value of the weight associated with the current power to the total regularization error
        magnituteMLSquared += abs((mL[j][0].item(0)))
    # Iterates through data points
    for i in range(0, len(xV)):
        # Value calculated from the maximum likelihood and current input
        wphi = 0
        # Multiply x^0...power by the weights
        for j in range(0, power + 1):
            wphi +=  (mL[j][0].item(0) * (xV[i] ** j))
        # Subtract the result from the target value and square it
        totalSoSError += ((yV[i] - wphi) ** 2)
    # Divide the total sum-of-squares error by the number of data points to get the *mean* sum-of-squares error
    totalSoSError = ((totalSoSError/ len(xV)) + ((lr) * magnituteMLSquared))
    return totalSoSError

# Defines a method to print a matrix in a readable fashion
def printCleanMatrix(matrix):
    # Iterates through row in matrix
    for row in matrix:
        # Prints string version of row, followed by space
        print(str(row) + " ")

# Defines a method for k-fold cross-validation for a given:
# input: the input file of data points
# output: an output file to print th results to
# k: the k value
# degree: the power
def kfoldcrossvalidate(input, output, k, degree):
    # Initializes lists to hold the input X and T (again, apologies for the mix-up!) data points
    inputX = list() 
    inputY = list()

    # Initializes lists to hold the training and holdout X and T values
    xValues = list()
    holdoutXValues = list()
    yValues = list()
    holdoutYValues = list()

    # Initializes lists to hold the regularized and unregularized error for the training and holdout data sets
    totalEListTraining = list()
    totalEWListTraining = list()
    totalEListHoldout = list()
    totalEWListHoldout = list()

    # Open input file
    with open(input, "r") as inputFile:
        # Read in file, line by line
        for fileLine in inputFile:
            # Split the line using the character ','
            lineValues = fileLine.split(",")
            # If there are at least two values on the line
            if len(lineValues) >= 2:
                # Append the first to X value list and the second to the T value list
                inputX.append(float(lineValues[0]))
                inputY.append(float(lineValues[1].replace("\n", "")))
    # Close the input file
    inputFile.close()

    # Iterate from 0...k
    for i in range(0, k):
        # Iterate through the datapoints
        for j in range(0, len(inputX)):
            # If the current data point is in the training set 
            # That is: its index + the index of the current iteration from 0...k is not divisible by k
            if (((j + i)%k) != 0):
                # Append X and T values to Training set lists
                xValues.append(inputX[j])
                yValues.append(inputY[j])
            # Otherwise, append X and T values to the Holdout set lists
            else:
                holdoutXValues.append(inputX[j])
                holdoutYValues.append(inputY[j])
        
        # Initialize regularized and unregularized error for the training and holdout sets for a given iteration k
        errorTList = list()
        errorTWList = list()

        errorHList = list()
        errorHWList = list()
    
        # Open the output file
        with open(output, "a") as o:
            # Print the current k iteration to the output file
            powerItr = 1
            lR = (math.e ** degree)
            kitr = "k = {0}"
            o.write(kitr.format(i))
            o.write("\n")

            # Iterate powers from 1...20
            while(powerItr <= 20):
                
                # Make maximum likelihood weights for given power, and input x values, t values and lambda
                weightsItr = makeWML(xValues, yValues, powerItr, lR)
                # Print the current power to the output file 
                o.write(str(powerItr))
                o.write("\n")

                # Print each of the weights to the output file
                for l in weightsItr:
                    for m in l:
                        o.write(str(m.item(0))) 
                        o.write("\n")

                # Calculate and add the sum-of-squares error for the training set to the appropriate list
                sosErrorTItr = sumOfSquaresError(xValues, yValues, powerItr, weightsItr)
                errorTList.append(sosErrorTItr)

                # Calculate and add the sum-of-squares error for the holdout set to the appropriate list
                sosErrorHItr = sumOfSquaresError(holdoutXValues, holdoutYValues, powerItr, weightsItr)
                errorHList.append(sosErrorHItr)

                # Calculate and add the regularized sum-of-squares error for the training set to the appropriate list
                sosErrorTWItr = sumOfSquaresErrorWithRegularization(xValues, yValues,powerItr, weightsItr, lR)
                errorTWList.append(sosErrorTWItr)

                # Calculate and add the regularized sum-of-squares error for the holdout set to the appropriate list
                sosErrorHWItr = sumOfSquaresErrorWithRegularization(holdoutXValues, holdoutYValues, powerItr, weightsItr, lR)
                errorHWList.append(sosErrorHWItr)

                # Increase the power
                powerItr += 1
        
                o.write("\n")
            
            o.write("\n\n")
        # Close the output file
        o.close()
        
        # Append the lists of regularized and unregularized sum-of-squares error for training and holdout datasets to the appropriate lists
        totalEListTraining.append(errorTList)
        totalEListHoldout.append(errorHList)

        totalEWListTraining.append(errorTWList)
        totalEWListHoldout.append(errorHWList)
    
    # Initialize lists to hold the average regularized and unregularized sum-of-squares error for training and holdout datasets
    averageTrainingError = [0] * 20
    averageTrainingErrorWithRegularization = [0] * 20

    averageHoldoutError = [0] * 20
    averageHoldoutErrorWithRegularization = [0] * 20

    # Iterate from 0...k
    for i in range(0, k):
        # Iterate through powers
        for p in range(0, 20):
            
            # Add 1/k * the sum-of-squares error for maximum likelihood weights for the current power for the training set 
            averageTrainingError[p] += (totalEListTraining[i][p] / k)
            # Add 1/k * the regularized sum-of-squares error for maximum likelihood weights for the current power for the training set 
            averageTrainingErrorWithRegularization[p] += (totalEWListTraining[i][p] / k)

            # Add 1/k * the sum-of-squares error for maximum likelihood weights for the current power for the holdout set 
            averageHoldoutError[p] += (totalEListHoldout[i][p] / k)
            # Add 1/k * the regularized sum-of-squares error for maximum likelihood weights for the current power for the holdout set 
            averageHoldoutErrorWithRegularization[p] += (totalEWListHoldout[i][p] / k)

    # Plot the average training and holdout sum-of-squares error over the powers
    plt.plot(np.arange(1, 21, 1), averageTrainingError, color='red')
    plt.plot(np.arange(1, 21, 1), averageHoldoutError, color='blue')

    # Set X and Y labels
    plt.xlabel('Power')
    plt.ylabel('Sum of Squares Error')

    # Set legend
    plt.legend(['Training Data', 'Holdout Data'])
    # Set x ticks
    plt.xticks(np.arange(1, 20, 1))

    # Set title
    plt.title('Variability in Sum of Squares Error with Regard to Power of Fit')

    # Show the plot
    plt.show()

    # Plot the average training and holdout regularized sum-of-squares error over the powers
    plt.plot(np.arange(1, 21, 1), averageTrainingErrorWithRegularization, color='red')
    plt.plot(np.arange(1, 21, 1), averageHoldoutErrorWithRegularization, color='blue')

    # Set legend
    plt.legend(['Training Data', 'Holdout Data'])
    # Set x scale
    plt.xticks(np.arange(1, 20, 1))

    # set x and y labels
    plt.xlabel('Power')
    plt.ylabel('Sum of Squares w/Regularization')

    # set title
    plt.title('Variability in Regularized Sum of Squares Error with Regard to Power of Fit')

    # show the graph
    plt.show()

# Define method to plot the unregularized sum-of-squares error for the maximum likelihood weights with a given polynomial over regularization constants e^-6...e^1, given:
# input: an input file with data points
# output: an output file to print to
# power: the power of polynomial to consider
def plotErrorOverE1toM6(input, output, power):
    # Initializes lists to hold the input X and T (again, apologies for the mix-up!) data points
    X = list()
    Y = list()

    # Open the input file
    with open(input, "r") as inputFile:
        # Read the file in line by line
        for fileLine in inputFile:
            # Split the current line by the character ','
            lineValues = fileLine.split(",")
            # If there are more than two variables on a line
            if len(lineValues) >= 2:
                # Append the first to the x values list and the next to the t values list
                X.append(float(lineValues[0]))
                Y.append(float(lineValues[1].replace("\n", "")))
    # Close the input file
    inputFile.close()

    # Define a list to hold the unregularized sum-of-squares error for all the possible values of lambda (e^-6...e^1)
    errorList = list()

    # Open output file
    with open(output, "w") as o:
        # Write the power
        pwrStr = "m = {0}"
        o.write(pwrStr.format(power))
        o.write("\n")
        # Iterate through all the possible powers of e in lambda
        for i in range(-6, 1):
            # Make the maxiumum likelihood weights for the given input values, target values, power, and lambda
            wML = makeWML(X, Y, power, (math.e ** i))
            # Generate the sum-of-squares error for the current maxiumum likelihood weights
            currentError = sumOfSquaresError(X, Y, power, wML)
            # Append that error to the errorlist
            errorList.append(currentError)
            # Print the error and power of e
            eItr = "{0} : "
            o.write(eItr.format(i))
            o.write(str(currentError))
            o.write("\n")
        o.write("\n")
    # Close the output file
    o.close()

    # Label the x and y axis
    plt.xlabel("Powers of E in Regularization Constant")  
    plt.ylabel("Sum of Squares Error in Maximum Likelihood Weights")
    # Label the title
    plt.title("Dependency of Error on Regularization Constant for Power %s" %str(power))
    # Plot the unregularizes sum-of-squares error vs. the power of e in the lambda
    plt.plot(np.arange(-6, 1, 1), errorList)

    # Show the graph
    plt.show()

# Define a method to plot the maximum likelihood solution for a particular power and lambda, given:
# input, an input file of data points
# output, an output file to write results to
# power, the power of the polynomial to fit to
# lr, the lambda value
def plotWML(input, output, power, lr):
    # Initializes lists to hold the input X and T (again, apologies for the mix-up!) data points
    X = list()
    Y = list()

    # Open the input file
    with open(input, "r") as inputFile:
        # Read in file line by line
        for fileLine in inputFile:
            # Split the current line by the character ','
            lineValues = fileLine.split(",")
            # If there are more than two variables on a line
            if len(lineValues) >= 2:
                # Append the first to the X value list and the second to the T value list
                X.append(float(lineValues[0]))
                Y.append(float(lineValues[1].replace("\n", "")))
    # Close the input file
    inputFile.close()

    # Open the output file
    with open(output, "a") as o:
        # Write the power 
        pwrStr = "{0}"
        o.write(pwrStr.format(power))
        o.write("\n")
        # Generate the maximum likelihood weights for the given X data points, T data points, power and lambda
        wML = makeWML(X, Y, power, lr)
        # Iterate through the weights and print them to the output value
        for l in wML:
            for m in l:
                o.write(str(m.item(0))) 
                o.write("\n")

        # Generate the unregularized sum-of-squares for the maximum likelihood weights
        currentError = sumOfSquaresError(X, Y, power, wML)
        
        eItr = "{0} : "
        o.write(eItr.format(lr))
        o.write(str(currentError))
        o.write("\n")
    # Close the output file
    o.close()

    # Get the values predicted by the x values and the maximum likelihood values
    yPredicted = np.dot(makePhi(X, power), wML)

    # Plot the data points
    plt.scatter(X, Y)
    # Plot the maximum likelihood polynomial fit
    plt.plot(X, yPredicted)

    # Set the title of the plot
    plt.title("Power %d Fit" %power)

    # Show the plot
    plt.show()

# Print informative message about possible choices for program
print("Pick a program functionality to utilize:")
print("1. Given a particular dataset, calculate and plot the maximum likelihood polynomial fit for a input power and regularization constant (input: '1')")
print("2. Given a particular dataset, calculate and plot the average regularized and unregularized sum-of-squares error using k-fold cross-validation for the polynomial fits of powers 1...20, with an input regularization constant (input: '2')")
print("3. Given a particular dataset, calculate and plot the sum of squares error over regularization constants e^-6...1, for an input polynomial (input: '3')")
choiceName = input("Input? ")

# Prompt for input and output files
inputName = input("Enter a file to read datapoints in from: ")
outputName = input("Enter a file to read information out to: ")

choiceNum = int(choiceName)

# If choice 1 (plot the maximum likelihood polynomial fit for a given power and regularization constant) was chosen...
if (choiceNum == 1):
    # Prompt for power and regularization constant
    powerName = input("Enter an integer value for the power of polynomial fit: ")
    lambdaName = input("Enter the value of the regularization constant: ")

    # Plot the maximum likelihood polynomial...
    plotWML(inputName, outputName, int(powerName), float(lambdaName))
# If choice 2 (k-fold cross-validation average regularized and unregularized error) was chosen
elif (choiceNum == 2):
    # Prompt for k value and regularization constant
    kName = input("Enter an integer value for k (suggested: 5): ")
    lambdaName = input("Enter the DEGREE of the regularization constant: ")

    # Run the k-fold cross-validation function
    kfoldcrossvalidate(inputName, outputName, int(kName), int(lambdaName))
# If choice 3 (plot sum-of-squares error for a given power over different regularization constants)
elif (choiceNum == 3):
    # Prompt for regularization constant
    powerName = input("Enter an integer value for the power of polynomial fit: ")
    
    # Plot error
    plotErrorOverE1toM6(inputName, outputName, int(powerName))
# Else, print error message
else:
    print("Could not parse choice...qutting program.")


# kfoldcrossvalidate(inputName, outputName, 5)

# Input value paths (on my computer)
inputFiles = ["C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/datasets/A", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/datasets/B", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/datasets/C", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/datasets/D", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/datasets/E"]

# Output value paths (on my computer)
outputFiles = ["C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/outputA.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/outputB.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/outputC.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/outputD.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/outputE.txt"]

# Output value paths (for lambda error, on my computer)
outputEFiles = ["C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/A1Error.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/B2Error.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/C3Error.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/D5Error.txt", "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/E7Error.txt"]

# Suspected powers
powerE = [1, 2, 3, 5, 7]

# for i in range(0,5):
    # plotErrorOverE1toM6(inputFiles[i], outputEFiles[i], powerE[i])

# plotErrorOverE1toM6(inputFiles[4], "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/E5Error.txt", 5)

# plotWML(inputFiles[4], "C:/Users/15854/Documents/UR/Year 4/Spring 2022/CSC 246/Project 1/EFit.txt", 7, (math.e ** (-6)))

# for i in range(0, 5):
    # kfoldcrossvalidate(inputFiles[i], outputFiles[i], 5, -4)





# LOTS AND LOTS OF UNUSUED CODE...

# plt.scatter(xValues, yValues, marker =".")
# plt.xlabel("Input (x)")
# plt.ylabel("Target Values (t)")
# plt.title("Linear Regression Dataset")

# plt.xlabel('Input Value')
# plt.ylabel('Target Value')
# plt.title('Training Data')

# SHOW FUNCTION
# plt.show()

# inputPowerAsStr = input("Enter a power of basis function to test:")
# inputPower = int(inputPowerAsStr)

# inputLearningRateAsString = input("Enter a regularization constant to test:")
# learningRate = float(inputLearningRateAsString)

# maxLikelihoodWeights = makeWML(xValues,yValues, inputPower, learningRate)

# print("The maximum likelihood weights are: ")
# printCleanMatrix(maxLikelihoodWeights)
# print(makePhi(xValues, inputPower))

# print("The value of the likelihood function for this power and regularization constant on the training values is:")
# maxBeta = beta(xValues, yValues, inputPower, maxLikelihoodWeights)
# print("maxBeta", maxBeta)
# likelihoodFunction = likelihood(xValues, yValues, maxLikelihoodWeights, inputPower, maxBeta)
# print(likelihoodFunction)

# print("The value of the sum of squares error for this power and regularization constant on the training values is:")
# sosError = sumOfSquaresError(xValues, yValues, inputPower, maxLikelihoodWeights)
# print(sosError)

# yPredicted = np.dot(makePhi(xValues, inputPower), maxLikelihoodWeights)

# plt.scatter(xValues, yValues)
# plt.plot(xValues, yPredicted)
# plt.show()

# print("The value of the likelihood function for this power and regularization constant on the holdout values is:")
# maxBetaHoldout = beta(holdoutXValues, holdoutYValues, inputPower, maxLikelihoodWeights)
# print("maxBeta", maxBetaHoldout)
# likelihoodFunctionHoldout = likelihood(holdoutXValues, holdoutYValues, maxLikelihoodWeights, inputPower, maxBetaHoldout)
# print(likelihoodFunctionHoldout)

# print("The value of the sum of squares error for this power and regularization constant on the holdout values is:")
# sosErrorHoldout = sumOfSquaresError(holdoutXValues, holdoutYValues, inputPower, maxLikelihoodWeights)
# print(sosErrorHoldout)

# yHoldoutPredicted = np.dot(makePhi(holdoutXValues, inputPower), maxLikelihoodWeights)


# plt.scatter(holdoutXValues, holdoutYValues)
# plt.plot(holdoutXValues, yHoldoutPredicted)
# plt.show()


#a = plt.figure()
# axes = a.add_axes([1, 20), ylim=(min(min(errorTList), min(errorHList)), max(max(errorTList), max(errorHList))))

# phi^j (x) = x^j

# print(xValues)
# print(yValues)

# C:\Users\15854\Documents\UR\Year 4\Spring 2022\CSC 246\Project 1\datasets\A