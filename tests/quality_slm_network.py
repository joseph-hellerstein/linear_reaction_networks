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
TIMES = np.linspace(0, 5, 50)
NUM_STAGE = 3
NUM_CHECK = 3
FRACTIONAL_DEVIATION = 0.1


#############################
# Tests
#############################
class TestSLMNetwork(unittest.TestCase):

    def setUp(self):
        self.network = self.makeNetwork()

    def makeNetwork(self, num_stage=NUM_STAGE, times=TIMES):
        ks = np.random.uniform(1, 100, num_stage)
        kps = np.random.uniform(1, 100, num_stage)
        return SLMNetwork.makeSequentialNetwork(ks, kps, times=times)

    def testConcatenate(self):
        #if IGNORE_TEST:
        #    return
        results = []
        for _ in range(NUM_CHECK):
            network = self.makeNetwork()
            cnetwork = self.network.concatenate(network)
            results.append(cnetwork.isValid(is_plot=True, fractional_deviation=FRACTIONAL_DEVIATION, times=TIMES))
        import pdb; pdb.set_trace()
        print(sum(results)/len(results))
        self.assertTrue(all(results))

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
  unittest.main()