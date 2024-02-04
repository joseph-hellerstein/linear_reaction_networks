"""Evaluations of quality of predictions for many random parameter values."""
from lrn_builder.slm_network import SLMNetwork # type: ignore
from lrn_builder.named_transfer_function import NamedTransferFunction   # type: ignore

import control # type: ignore
import numpy as np
import unittest
import tellurium as te # type: ignore

s = control.TransferFunction.s


IGNORE_TEST = True
IS_PLOT = False
TIMES = np.linspace(0, 10, 100)
NUM_STAGE = 5
NUM_CHECK = 10


#############################
# Tests
#############################
class TestSLMNetwork(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.network = self.makeNetwork()

    def makeNetwork(self, num_stage=NUM_STAGE, times=TIMES):
        ks = np.random.uniform(0.1, 1, num_stage)
        kps = np.random.uniform(0.1, 1, num_stage)
        return SLMNetwork.makeSequentialNetwork(ks, kps, times=times)

    def testConcatenate(self):
        if IGNORE_TEST:
            return
        for _ in range(NUM_CHECK):
            network = self.makeNetwork()
            cnetwork = self.network.concatenate(network)
            self.assertTrue(cnetwork.isValid(is_plot=IS_PLOT))

    def testBranchjoin(self):
        if IGNORE_TEST:
            return
        for _ in range(NUM_CHECK):
            network = self.makeNetwork()
            bjn = self.network.branchjoin(network)
            self.assertTrue(bjn.isValid(is_plot=IS_PLOT))
       

if __name__ == '__main__':
  unittest.main()