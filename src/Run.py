#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    myMLP = MultilayerPerceptron(data.trainingSet,
                                 data.validationSet,
                                 data.testSet,
                                 layers=[512, 256, 128],
                                 learningRate=0.01,
                                 weightDecayRate=0.0005,
                                 epochs=100,
                                 loss='crossentropy',
                                 inputActivation='lrelu',
                                 outputActivation='softmax')
                                        
    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMLP has been training..")
    myMLP.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    MLPPred = myMLP.evaluate()

    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the mlp recognizer:")
    #evaluator.printComparison(data.testSet, MLPPred)
    evaluator.printAccuracy(data.testSet, MLPPred)

    # Draw
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(myMLP.performancesTraining,
                                myMLP.performancesValidation, myMLP.epochs)
    
    
if __name__ == '__main__':
    main()
