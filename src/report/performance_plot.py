import matplotlib.pyplot as plt


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, performancesTraining, performancesValidation,
                               epochs, learningRate=0, weightDecayRate=0, show=True):
        plt.plot(range(epochs), performancesTraining, 'k',
                 range(epochs), performancesTraining, 'bo')
        plt.plot(range(epochs), performancesValidation, 'k',
                 range(epochs), performancesValidation, 'ro')
        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")

        if show:
            plt.show()
        else:
            plt.savefig("../plots/Learning Rate " + str(learningRate)
                    + ", Weight Decay Rate " + str(weightDecayRate)
                    + ".png")
        plt.clf()
