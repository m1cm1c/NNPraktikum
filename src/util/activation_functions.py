# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np

class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        # use e^x from numpy to avoid overflow
        return 1/(1+np.exp(-1.0*netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return netOutput * (1.0 - netOutput)

    @staticmethod
    def tanh(netOutput):
        # return 2*Activation.sigmoid(2*netOutput)-1
        ex = np.exp(1.0*netOutput)
        exn = np.exp(-1.0*netOutput)
        return np.divide(ex-exn, ex+exn)  # element-wise division

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        return (1-Activation.tanh(netOutput)**2)

    @staticmethod
    def rectified(netOutput):
        # leaky relu as normal relu seems to be too aggressive
        # and easily kills the neurons
        return np.asarray([max(0.00, i) for i in netOutput])

    @staticmethod
    def rectifiedPrime(netOutput):
        # reluPrime=1 if netOutput > 0 otherwise 0
        return netOutput > 0

    @staticmethod
    def leakyRectified(netOutput):
        # leaky relu as normal relu seems to be too aggressive
        # and easily kills the neurons
        return np.asarray([max(0.01*i, i) for i in netOutput])

    @staticmethod
    def leakyRectifiedPrime(netOutput):
        # leaky relu as normal relu seems to be too aggressive
        # and easily kills the neurons
        return np.asarray([1.0 if out > 0 else 0.01 for out in netOutput])

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # identityPrime = 1
        return ones(netOutput.size)

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        # exps = [np.exp(out) for out in netOutput]
        # update for numerical stability (avoid overflows)
        exps = [np.exp(out - np.max(netOutput)) for out in netOutput]
        return exps / np.sum(exps)
        
    @staticmethod
    def softmaxPrime(netOutput):
        #https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        jacobian_m = np.diag(netOutput)

        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = netOutput[i] * (1-netOutput[i])
                else: 
                    jacobian_m[i][j] = -netOutput[i]*netOutput[j]

        return jacobian_m
        
    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        elif str == 'lrelu':
            return Activation.leakyRectified
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'softmax':
            return Activation.softmaxPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        elif str == 'lrelu':
            return Activation.leakyRectifiedPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
