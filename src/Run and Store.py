#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

import os

from joblib import Parallel, delayed
import multiprocessing


def main():
    if not os.path.exists("../plots"):
        os.makedirs("../plots")

    numCores = 8
    Parallel(n_jobs=numCores)(delayed(process)(learningRate, weightDecayRate)
                              for learningRate in [.1, .01, .001, .0001]
                              for weightDecayRate in [.1, .01, .001, .0001, 0])

def process(learningRate, weightDecayRate):
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      oneHot=False)
    myMLP = MultilayerPerceptron(data.trainingSet,
                                 data.validationSet,
                                 data.testSet,
                                 learningRate=learningRate,
                                 weightDecayRate=weightDecayRate,
                                 epochs=50,
                                 loss='crossentropy',
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
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()
    # lrPred = myLRClassifier.evaluate()
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
                                myMLP.performancesValidation,
                                myMLP.epochs,
                                learningRate,
                                weightDecayRate)

if __name__ == '__main__':
    main()
