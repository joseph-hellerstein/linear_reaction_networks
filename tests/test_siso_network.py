from netapprox.siso_network import SISONetwork # type: ignore

import control # type: ignore
import numpy as np
import pandas as pd
import re
import unittest
import tellurium as te # type: ignore


IGNORE_TEST = True
IS_PLOT = True
LINEAR_MDL = """
model *main_model()
species S1, S2
S1 -> S2; k1*S1
J2: S2 -> ; k2*S2
k1 = 1
k2 = 2
S1 = 10
S2 = 0
end
"""


#############################
# Tests
#############################
class TestSISONetwork(unittest.TestCase):

    def setUp(self):
        k1 = 1
        k2 = 2
        tf = control.TransferFunction([k1], [1, k2])
        self.network = SISONetwork(LINEAR_MDL, "S1", "S2", k1, k2, tf)

    def check(self, network=None):
        if network is None:
            network = self.network
        antimony_str = network.getAntimony()
        rr = te.loada(str(network))
        data = rr.simulate(0,20, 2000, selections=["time", "S1", "S2", "S3"])
        self.assertTrue(len(data) > 0)
        if IS_PLOT:
            rr.plot()
        return data

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.network, SISONetwork))

    def testCopyAndEqual(self):
        if IGNORE_TEST:
            return
        self.init()
        builder = self.builder.copy()
        self.assertTrue(builder == self.builder)
        #
        builder.makeBoundarySpecies("S1")
        self.assertFalse(builder == self.builder)

    def testGetAntimony(self):
        if IGNORE_TEST:
            return
        antimony_str = self.network.getAntimony()
        self.assertEqual(antimony_str, LINEAR_MDL)
        self.network.template.isValidAntimony()

    def testPlotStaircaseResponse(self):
        if IGNORE_TEST:
            return
        self.network.plotStaircaseResponse(is_plot=IS_PLOT)

    def testMakeTwoSpeciesNetwork(self):
        #if IGNORE_TEST:
        #    return
        kI = 0.5
        kO = 1.0
        times = np.linspace(0, 10, 100)
        network = SISONetwork.makeTwoSpeciesNetwork(kI, kO, times=times)
        self.assertTrue(isinstance(network, SISONetwork))
        self.assertTrue(network.input_name == "SI")
        self.assertTrue(network.output_name == "SO")
        self.assertTrue(network.kI == kI)
        self.assertTrue(network.kO == kO)
        si = 10
        response_df = network.simulate(kIO=1, kO=0.5, SI=si)
        self.assertEqual(response_df.loc[0, "SI"], si)
        self.assertGreater(response_df.loc[len(times)-1, "SO"], 0.1)
       

if __name__ == '__main__':
  unittest.main()