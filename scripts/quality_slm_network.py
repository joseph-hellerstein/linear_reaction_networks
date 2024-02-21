"""Evaluations of quality of predictions for many random parameter values."""
from lrn_builder.slm_network import SLMNetwork # type: ignore
from lrn_builder.named_transfer_function import NamedTransferFunction   # type: ignore

import numpy as np
import matplotlib.pyplot as plt


IS_DONT = False
TIMES = np.linspace(0, 10, 100)
NUM_STAGE = 3
NUM_CHECK = 30
FRACTIONAL_DEVIATION = 0.01
RESULT_DCT:dict = {}

class QualityAnalyzer(object):

    def __init__(self, num_stage=NUM_STAGE, num_check=NUM_CHECK, times=TIMES):
        self.num_stage = num_stage
        self.num_check = num_check
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
        nbin = max(int(self.num_check/5), 2)
        nbin = 100
        bins = np.linspace(0, 1, nbin)
        for method in result_dct.keys():
            if len(result_dct[method]) == 0:
                continue
            _, ax = plt.subplots(1)
            ax.hist(result_dct[method], bins=bins, label=method)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, len(result_dct[method])])
            ax.set_xlabel("score")
            ax.set_ylabel("count")
            ax.set_title(method[2:])
        plt.show()

    def doConcatenate(self):
        """
        Tests concatenate of networks
        """
        results = []
        if IS_DONT:
            return results
        for _ in range(self.num_check):
            network = self.makeNetwork()
            cnetwork = self.network.concatenate(network)
            _, builder = cnetwork.plotStaircaseResponse(is_plot=False, times=TIMES)
            _, score = cnetwork.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            results.append(score)
        return results

    def doBranchjoin(self):
        results = []
        if IS_DONT:
            return results
        for _ in range(self.num_check):
            network = self.makeNetwork()
            k1a, k1b, k2a, k2b, k3 = np.random.uniform(0, 10, 5)
            bjn = self.network.branchjoin(network, k1a=k1a, k1b=k1b, k2a=k2a, k2b=k2b, k3=k3)
            _, builder = bjn.plotStaircaseResponse(is_plot=False, times=TIMES)
            df, score = bjn.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            results.append(score)
        return results
    
    def doScale(self):
        results = []
        if IS_DONT:
            return results
        for _ in range(self.num_check):
            network = self.makeNetwork()
            k1, k2, scale = np.random.uniform(0, 10, 3)
            network = self.network.scale(m=scale, k1=k1, k2=k2)
            _, builder = network.plotStaircaseResponse(is_plot=False, times=TIMES)
            df, score = network.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            results.append(score)
        return results
    
    def doPfeedback(self):
        results = []
        if IS_DONT:
            return results
        for _ in range(self.num_check):
            network = self.makeNetwork()
            k1, k2, k3, k4, k5 = np.random.uniform(0, 100, 5)
            network = self.network.pfeedback(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)
            _, builder = network.plotStaircaseResponse(is_plot=False, times=TIMES)
            df, score = network.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            import pdb; pdb.set_trace()
            results.append(score)
        return results
    
    def doNfeedback(self):
        results = []
        if IS_DONT:
            return results
        for _ in range(self.num_check):
            network = self.makeNetwork()
            k1, k2, k3, k4, k5 = np.random.uniform(0, 100, 5)
            network = self.network.nfeedback(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)
            _, builder = network.plotStaircaseResponse(is_plot=False, times=TIMES)
            df, score = network.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                                 fractional_deviation=FRACTIONAL_DEVIATION)
            import pdb; pdb.set_trace()
            results.append(score)
        return results
       

if __name__ == '__main__':
  quality_analyzer = QualityAnalyzer()
  quality_analyzer.analyze()