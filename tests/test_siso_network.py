from netapprox.siso_network import SISONetwork # type: ignore

import control # type: ignore
import controlSBML as ctl # type: ignore
import numpy as np
import pandas as pd
import re
import unittest
import tellurium as te # type: ignore


IGNORE_TEST = False
IS_PLOT = False
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
LINEAR_MDL1 = """
model *main_model()
species S1, S2
S1 -> S2; k1*S1
J2: S2 -> ; k2*S2
k1 = 1
k2 = 2
S1 = 0
S2 = 0
end
"""
TIMES = np.linspace(0, 10, 100)


#############################
# Tests
#############################
class TestSISONetwork(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.init()

    def init(self, model=LINEAR_MDL, times=TIMES):
        k1 = 1
        k2 = 2
        tf = control.TransferFunction([k1], [1, k2])
        self.network = SISONetwork(model, "S1", "S2", k1, k2, tf, times=times)

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
        network = self.network.copy()
        self.assertTrue(network == self.network)

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
        if IGNORE_TEST:
            return
        self.init()
        kI = 0.5
        kO = 1.0
        times = np.linspace(0, 30, 300)
        network = SISONetwork.makeTwoSpeciesNetwork(kI, kO, times=times)
        self.assertTrue(isinstance(network, SISONetwork))
        self.assertTrue(network.input_name == "SI")
        self.assertTrue(network.output_name == "SO")
        self.assertTrue(network.kI == kI)
        self.assertTrue(network.kO == kO)
        si = 10
        response_ts, _ = network.plotStaircaseResponse(is_plot=IS_PLOT, times=times)
        indices = list(response_ts.index)
        end_index = indices[-1]
        so = network.transfer_function.dcgain()*si
        self.assertTrue(np.isclose(response_ts.loc[end_index, "SI_staircase"], si))
        self.assertGreaterEqual(so, response_ts.loc[end_index, "SO"])

    def testPlotTransferFunction(self):
        if IGNORE_TEST:
            return
        timeseries = self.network.plotTransferFunction(is_plot=IS_PLOT)
        self.assertTrue(isinstance(timeseries, ctl.Timeseries))
        timeseries = self.network.plotTransferFunction(is_simulation=False, is_plot=IS_PLOT)
        self.assertTrue(isinstance(timeseries, ctl.Timeseries))

    def testConcatenate(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1)
        network = self.network.copy()
        cnetwork = self.network.concatenate(network)
        self.assertTrue(cnetwork.input_name == "SI")
        self.assertTrue(cnetwork.output_name == "SO")
        # Do simulations
        self.assertTrue(cnetwork.isValid())

    def testConcatenate2(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 100, 1000))
        network = self.network.copy()
        cnetwork = network.concatenate(self.network)
        for _ in range(20):
            cnetwork = cnetwork.concatenate(self.network)
        self.assertTrue(cnetwork.input_name == "SI")
        self.assertTrue(cnetwork.output_name == "SO")
        # Do simulations
        self.assertTrue(cnetwork.isValid())

    def testMakeSequentialNetwork(self):
        if IGNORE_TEST:
            return
        network = SISONetwork.makeSequentialNetwork([1, 2, 3], [0.5, 0.6, 0.7])
        self.assertTrue(network.input_name == "S0")
        self.assertTrue(network.output_name == "S3")
        _ = network.plotTransferFunction(is_plot=IS_PLOT)
        self.assertTrue(network.isValid())
       

if __name__ == '__main__':
  unittest.main()