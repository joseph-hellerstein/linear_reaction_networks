"""
Evaluations of quality of predictions for many random parameter values.

TODO:
  1. Make runnable script
  2. unittests
  3. progress report
"""
from lrn_builder.slm_network import SLMNetwork # type: ignore
from lrn_builder.named_transfer_function import NamedTransferFunction   # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple, Dict


TIMES = np.linspace(0, 10, 100)
NUM_STAGE = 3
NUM_CHECK = 30
FRACTIONAL_DEVIATION = 0.01
KMAX = 10.0  # Maximum value for a parameter

class AbstractOperator(object):
    """
    Abstraction for operators that provides a way to simplify their use.
    """
    def __init__(self, operator_name, is_unary=True, num_parameter:Optional[int]=None, parameter_names:Optional[List[str]]=None,
                 kmax:Optional[float]=KMAX):
        self.operator_name = operator_name
        self.is_unary = is_unary
        if num_parameter is not None:
            self.parameter_names = ["k%d" % n for n in range(1, num_parameter+1)]
        elif parameter_names is not None:
            self.parameter_names = parameter_names
        else:
            self.parameter_names = []
        self.kmax = kmax

    @staticmethod 
    def makeNetwork(times=TIMES, num_stage=NUM_STAGE)->SLMNetwork:
        ks = np.random.uniform(1, 100, num_stage)
        kps = np.random.uniform(1, 100, num_stage)
        return SLMNetwork.makeSequentialNetwork(ks, kps, times=times)

    def do(self, network:SLMNetwork)->Tuple[SLMNetwork, float]:
        """
        Executes the operator on the network and evaluates the result.

        Args:
            network: network A to be modified
        Returns:
            network: new network
            score: float
        """
        # Construct the arguments
        if self.is_unary:
            args = []
        else:
            args = [self.makeNetwork()]
        # Construct the arguments
        kwargs = {k: np.random.uniform(0, self.kmax) for k in self.parameter_names}  # type: ignore
        # Run the operator
        method = getattr(network, self.operator_name)
        new_network = method(*args, **kwargs)
        # Score the result
        _, builder = new_network.plotStaircaseResponse(is_plot=False, times=TIMES)
        _, score = new_network.named_transfer_function.evaluate(str(builder), is_plot=False,
                                                            fractional_deviation=FRACTIONAL_DEVIATION)
        return network, score

    @classmethod
    def makeOperators(cls)->dict:
        """
        Makes descriptions
        """
        return {"concatenate": cls("concatenate", is_unary=False),
                "branchjoin": cls("branchjoin", is_unary=False, parameter_names=["k1a", "k1b", "k2a", "k2b", "k3"]),
                "scale": cls("scale", is_unary=True, num_parameter=2),
                "pfeedback": cls("pfeedback", is_unary=True, num_parameter=5),
                "nfeedback": cls("nfeedback", is_unary=True, num_parameter=5)}


class SingleOperatorQualityAnalyzer(object):
    # Analyzes the quality of a single operator by comparing the simulation with the transfer function
    # different networks with different parameters for the operator.

    def __init__(self, num_stage=NUM_STAGE, num_check=NUM_CHECK, times=TIMES):
        self.num_stage = num_stage
        self.num_check = num_check
        self.times = times
        self.operator_dct = AbstractOperator.makeOperators()
        self.operator_names = self.operator_dct.keys()
    
    def analyze(self, is_plot:bool=True)->Dict[str, List[float]]:
        """
        Analyzes the quality of the predictions.

        Args:
            is_plot (bool): whether to plot the results
        Returns:
            dict:
                key: operator name
                value: list of scores
        """
        result_dct:dict = {n: [] for n in self.operator_names}
        for operator_name in self.operator_names:
            for _ in range(self.num_check):
                network = AbstractOperator.makeNetwork()
                _, score = self.operator_dct[operator_name].do(network)
                result_dct[operator_name].append(score)
        # Do the plots
        if is_plot:
            for operator_name, scores in result_dct.items():
                self.plot(operator_name, scores)
            plt.show()
        return result_dct

    @staticmethod
    def plot(analysis_name:str, scores:List[float]):
        """
        Plots histogram of scores.

        Args:
            analysis_name (str): name of the analysis conducted
            scores (_type_): _description_
        """
        if len(scores) == 0:
            return
        nbin = 100
        bins = np.linspace(0, 1, nbin)
        _, ax = plt.subplots(1)
        ax.hist(scores, bins=bins, label=analysis_name)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, len(scores)])
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_title(analysis_name)


class PairwiseQualityAnalyzer(object):
    # Considers methods in combination

    def __init__(self, num_stage:int=NUM_STAGE, num_check:int=NUM_CHECK, times=TIMES):
        self.num_stage = num_stage
        self.num_check = num_check
        self.times = times
        self.operator_dct = AbstractOperator.makeOperators()
        self.method_names = self.operator_dct.keys()

    def analyze(self, is_plot:bool=True)->Dict[str, List[float]]:
        """
        Analyzes the quality of the predictions.

        Args:
            is_plot (bool): whether to plot the results
        Returns:
            dict:
                key: operator name
                value: list of scores
        """
        result_dct:dict = {}
        for op1 in self.method_names:
            for op2 in self.method_names:
                entry = "%s-%s" % (op1, op2)
                result_dct[entry] = []
                for _ in range(self.num_check):
                    network = AbstractOperator.makeNetwork()
                    composed_network, _ = self.operator_dct[op1].do(network)
                    _, score = self.operator_dct[op2].do(composed_network)
                    result_dct[entry].append(score)
        # Do the plots
        if is_plot:
            for entry, scores in result_dct.items():
                SingleOperatorQualityAnalyzer.plot(entry, scores)
            plt.show() 
        #
        return result_dct
        

if __name__ == '__main__':
  sanalyzer = SingleOperatorQualityAnalyzer()
  sanalyzer.analyze()
  panalyzer = PairwiseQualityAnalyzer()
  panalyzer.analyze()