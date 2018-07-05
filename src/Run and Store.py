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
                              for learningRate in [0.0025, 0.005, 0.01, 0.02,
                                                   0.04, 0.08, 0.16]
                              for weightDecayRate in [0.000001, 0.000002,
                                                      0.000004, 0.000008,
                                                      0.000016, 0.000032,
                                                      0.000064, 0.000128,
                                                      0.000256, 0.000512])

def process(learningRate, weightDecayRate):
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      oneHot=False)
    myMLP = MultilayerPerceptron(data.trainingSet,
                                 data.validationSet,
                                 data.testSet,
                                 learningRate=learningRate,
                                 weightDecayRate=weightDecayRate,
                                 epochs=500 ,
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
