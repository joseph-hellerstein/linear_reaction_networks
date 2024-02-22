from scripts.quality_slm_network import AbstractOperator, SingleOperatorQualityAnalyzer, PairwiseQualityAnalyzer
from lrn_builder.slm_network import SLMNetwork  # type: ignore
from lrn_builder import constants as cn # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import tellurium as te # type: ignore
import unittest


IGNORE_TEST = True
IS_PLOT = True
MODEL_NAME = "a_model"
TWO_SPECIES_MDL = """
model *main_model1()
A -> B; kA1*A
kA1 = 1
A = 10
B = 0
end
// Extra text that follows the model
"""
LINEAR_MDL = """
model *%s ()
S1 -> S2; k1*S1
J1: S2 -> S3; k2*S2
J2: S3 -> S2; k3*S3
J3: S2 -> ; k4*S2

k1 = 1
k2 = 2
k3 = 3
k4 = 4
S1 = 10
S2 = 0
S3 = 0
end
""" % MODEL_NAME
BARE_MDL = """
S1 -> S2; k1*S1
J1: S2 -> S3; k2*S2
J2: S3 -> S2; k3*S3
J3: S2 -> ; k4*S2

k1 = 1
k2 = 2
k3 = 3
k4 = 4
S1 = 10
S2 = 0
S3 = 0
"""


#############################
# Tests
#############################
class TestAbstractOperator(unittest.TestCase):

    def setUp(self):
        self.abstract_operator = AbstractOperator("branchjoin", is_unary=False,
            parameter_names=["k1a", "k1b", "k2a", "k2b", "k3"])
        self.network = self.abstract_operator.makeNetwork(num_stage=3)
        
    def check(self, network:SLMNetwork):
        self.assertTrue(isinstance(network, SLMNetwork))
        self.assertTrue(network.named_transfer_function is not None)
        df, antimony_str, score = network.debug()
        self.assertTrue(score > 0.8)
        
    def test_makeNetwork(self):
        if IGNORE_TEST:
            return
        network = self.abstract_operator.makeNetwork(num_stage=5)
        self.check(network)
        self.assertTrue("S5 -" in network.getAntimony())

    def testDo(self):
        if IGNORE_TEST:
            return
        network, score = self.abstract_operator.do(self.network)
        self.check(network)
        self.assertTrue(isinstance(score, float))

    def testMakeOperators(self):
        if IGNORE_TEST:
            return
        dct = AbstractOperator.makeOperators()
        self.assertTrue(len(dct) > 0)
        for k, v in dct.items():
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, AbstractOperator))


class TestSingleOperatorQualityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SingleOperatorQualityAnalyzer(num_stage=1, num_check=2, times=np.linspace(0, 2, 20))

    def testAnalyze(self):
        if IGNORE_TEST:
            return
        result_dct = self.analyzer.analyze(is_plot=IS_PLOT)
        for key, value in result_dct.items():
            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(value, list))
            self.assertTrue(len(value) > 0)
            for v in value:
                self.assertTrue(isinstance(v, float))

    def testPlot(self):
        #if IGNORE_TEST:
        #    return
        if IS_PLOT:
            SingleOperatorQualityAnalyzer.plot("test", [0.1, 0.2, 0.3, 0.4, 0.5])
            plt.show()


class TestPairwiseQualityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = PairwiseQualityAnalyzer(num_stage=1, num_check=2, times=np.linspace(0, 1, 10))

    def testAnalyze(self):
        if IGNORE_TEST:
            return
        result_dct = self.analyzer.analyze(is_plot=IS_PLOT)
        for key, value in result_dct.items():
            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(value, list))
            self.assertTrue(len(value) > 0)
            for v in value:
                self.assertTrue(isinstance(v, float))


if __name__ == '__main__':
  unittest.main()