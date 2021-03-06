
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', inputActivation='sigmoid',
                 outputActivation='softmax', loss='bce',
                 learningRate=0.01, weightDecayRate=0, earlyStoppingEpochs=5,
                 epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        inputActivation : string
        outputActivation : string
        learningRate : float
        weightDecayRate : float
        earlyStopping : positive int
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        inputActivation : string
        outputActivation : string
        learningRate : float
        weightDecayRate : float
        epochs : positive int
        performancesTraining: array of floats
        performancesValidation: array of floats
        """

        self.learningRate = learningRate
        self.weightDecayRate = weightDecayRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.inputActivation = inputActivation
        self.outputActivation = outputActivation
        self.earlyStoppingEpochs = earlyStoppingEpochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'crossentropy':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + loss)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performancesTraining = []
        self.performancesValidation = []
        self.layers = []

        # Build up the network from specific layers
        if not layers:
            # Input layer
            self.layers.append(LogisticLayer(train.input.shape[1], 128,
                            None, self.inputActivation, False))

            # Output layer
            self.layers.append(LogisticLayer(128, 10,
                            None, self.outputActivation, True))

        else:
            nIn = train.input.shape[1]

            for layer in layers:
                self.layers.append(LogisticLayer(nIn, layer,
                            None, self.inputActivation, False))
                nIn = layer

            self.layers.append(LogisticLayer(nIn, 10,
                        None, self.outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1, axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        input = inp

        for layer in self.layers:
            if layer != self._get_input_layer():
                input = np.insert(input, 0, 1)

            input = layer.forward(input)

        return input

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """

        tempWeights = None
        tempDerivatives = None

        for layer in reversed(self.layers):
            if layer == self._get_output_layer():
                layer.computeDerivative(self.loss.calculateDerivative(target, layer.outp), 1.0)
            else:
                layer.computeDerivative(tempDerivatives, tempWeights[1:])

            tempWeights = layer.weights
            tempDerivatives = layer.deltas

        output = self._get_output_layer().outp
        return self.loss.calculateError(target, output)
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """

        for layer in reversed(self.layers):
            layer.updateWeights(self.learningRate, self.weightDecayRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):

            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            accuracyTraining = accuracy_score(self.trainingSet.label,
                                        self.evaluate(self.trainingSet))
            accuracyValidation = accuracy_score(self.validationSet.label,
                                        self.evaluate(self.validationSet))
            # Record the performance of each epoch for later usages
            # e.g. plotting, reporting..
            self.performancesTraining.append(accuracyTraining)
            self.performancesValidation.append(accuracyValidation)

            if verbose:
                print("Accuracy on training: {0:.2f}%"
                      .format(accuracyTraining * 100))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracyValidation * 100))
                print("-----------------------------")

            if epoch > self.earlyStoppingEpochs:
                noImprovement = 0

                for i in range(1, self.earlyStoppingEpochs+1):
                    currentPerf = self.performancesValidation[-i]
                    prevPerf = self.performancesValidation[-(i+1)]

                    if (currentPerf - prevPerf) < 0 or np.isclose(currentPerf, prevPerf, atol=0.001):
                        noImprovement += 1

                if noImprovement >= self.earlyStoppingEpochs:
                    print("Early stopping, no improvement in validation set.")

                    # resize epochs
                    self.epochs = epoch + 1

                    break


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)
        return np.argmax(outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))        

    def _train_one_epoch(self):
        for label, data in zip(self.trainingSet.label, self.trainingSet.input):
            self._feed_forward(data)
            self._compute_error(self._get_encoded_label(label))
            self._update_weights(self.learningRate)

    def _get_encoded_label(self, label):
        zeros = np.zeros(self._get_output_layer().nOut)
        zeros[label] = 1.0
        return zeros

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0, axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
