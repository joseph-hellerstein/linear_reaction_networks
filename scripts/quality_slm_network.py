"""Evaluations of quality of predictions for many random parameter values."""
from lrn_builder.slm_network import SLMNetwork # type: ignore
from lrn_builder.named_transfer_function import NamedTransferFunction   # type: ignore

import numpy as np
import matplotlib.pyplot as plt


IGNORE_TEST = True
IS_PLOT = False
TIMES = np.linspace(0, 10, 100)
NUM_STAGE = 3
NUM_CHECK = 30
FRACTIONAL_DEVIATION = 0.01
RESULT_DCT:dict = {}

class QualityAnalyzer(object):

    def __init__(self, num_stage=NUM_STAGE, times=TIMES):
        self.num_stage = num_stage
        self.times = times
        self.network = self.makeNetwork()

    def makeNetwork(self):
        ks = np.random.uniform(1, 100, self.num_stage)
        kps = np.random.uniform(1, 100, self.num_stage)
        return SLMNetwork.makeSequentialNetwork(ks, kps, times=self.times)
    
    def analyze(self):
        result_dct = {}
        methods = [m for m in dir(self) if m[0:2] == "do"]
        for method in methods:
            statement = "self.%s()" % method
            result_dct[method] = eval(statement)
        nbin = int(NUM_CHECK/5)
        bins = np.linspace(0, 1, nbin)
        for method in result_dct.keys():
            _, ax = plt.subplots(1)
            ax.hist(result_dct[method], bins=bins, label=method)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, len(result_dct[method])])
            ax.set_xlabel("score")
            ax.set_ylabel("count")
        plt.show()

    def doConcatenate(self):
        """
        Tests concatenate of networks
        """
        results = []
        for _ in range(NUM_CHECK):
            network = self.makeNetwork()
            cnetwork = self.network.concatenate(network)
            _, builder = cnetwork.plotStaircaseResponse(is_plot=False, times=TIMES)
            _, score = cnetwork.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            plt.close()
            results.append(score)
        return results

    def testBranchjoin(self):
        if IGNORE_TEST:
            return
        results = []
        for _ in range(NUM_CHECK):
            network = self.makeNetwork()
            bjn = self.network.branchjoin(network)
            results.append(cnetwork.isValid(is_plot=IS_PLOT, fractional_deviation=FRACTIONAL_DEVIATION, times=TIMES))
        print(sum(results)/len(results))
        self.assertTrue(all(results))
       

if __name__ == '__main__':
  quality_analyzer = QualityAnalyzer()
  quality_analyzer.analyze()