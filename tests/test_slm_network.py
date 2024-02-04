from lrn_builder.slm_network import SLMNetwork # type: ignore
from lrn_builder.named_transfer_function import NamedTransferFunction   # type: ignore

import control # type: ignore
import numpy as np
import unittest
import tellurium as te # type: ignore

s = control.TransferFunction.s


IGNORE_TEST = False
IS_PLOT = False
FRACTIONAL_DEVIATION = 0.02
SCORE_THRESHOLD = 0.95
FEEDBACK_MDL = """
// Created by libAntimony v2.14.0
model *main_model()

  // Compartments and Species:
  species SAI, SAO, SI, SO, XI;

  // Reactions:
  A__J0: SAI -> SAO; A_kIO*SAI;
  A__J1: SAO -> ; A_kO*SAO;
  _J0: SI -> XI; k1*SI;
  _J1: XI -> SAI; k2*XI;
  _J2: SAO -> SO; k3*SAO;
  _J3: SO -> ; k4*SO;
  _J4: SO -> XI; k5*SO;

  // Species initializations:
  SAI = 0;
  SAO = 0;
  SI = ;
  SO = ;
  XI = ;

  // Variable initializations:
  A_kIO = 1;
  A_kO = 1;
  k1 = 1;
  k2 = 1;
  k3 = 1;
  k4 = 1;
  k5 = 1;

  // Other declarations:
  const A_kIO, A_kO, k1, k2, k3, k4, k5;

//vvvvvvvvvAdded by ControlSBMLvvvvvvvvvv
const SI

// Staircase: SI->SO
SI = 0.000000
at (time>= 0.0): SI = 0.0
at (time>= 1.6616616616616617): SI = 2.0
at (time>= 3.3233233233233235): SI = 4.0
at (time>= 4.984984984984985): SI = 6.0
at (time>= 6.646646646646647): SI = 8.0
at (time>= 8.308308308308309): SI = 10.0
//^^^^^^^^^Added by ControlSBML^^^^^^^^^^
end
"""
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
class TestSLMNetwork(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.init()

    def init(self, model=LINEAR_MDL, times=TIMES):
        k1 = 1
        k2 = 2
        tf = control.TransferFunction([k1], [1, k2])
        self.network = SLMNetwork(model, "S1", "S2", k1, k2, tf, times=times)

    def check(self, network=None):
        if network is None:
            network = self.network
        rr = te.loada(str(network))
        data = rr.simulate(0,20, 2000, selections=["time", "S1", "S2", "S3"])
        self.assertTrue(len(data) > 0)
        if IS_PLOT:
            rr.plot()
        return data

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.network, SLMNetwork))

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
        network = SLMNetwork.makeTwoSpeciesNetwork(kI, kO, times=times)
        self.assertTrue(isinstance(network, SLMNetwork))
        self.assertTrue(network.input_name == "SI")
        self.assertTrue(network.output_name == "SO")
        self.assertTrue(network.kI == kI)
        self.assertTrue(network.kO == kO)
        si = 10
        response_ts, _ = network.plotStaircaseResponse(is_plot=IS_PLOT, times=times)
        indices = list(response_ts.index)
        end_index = indices[-1]
        so = network.named_transfer_function.transfer_functions[0].dcgain()*si
        self.assertTrue(np.isclose(response_ts.loc[end_index, "SI_staircase"], si))
        self.assertGreaterEqual(so, response_ts.loc[end_index, "SO"])

    def testConcatenate(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 100, 1000))
        network = self.network.copy()
        cnetwork = self.network.concatenate(network)
        self.assertTrue(cnetwork.input_name == "SI")
        self.assertTrue(cnetwork.output_name == "SO")
        # Do simulations
        self.assertTrue(cnetwork.isValid(is_plot=IS_PLOT,
                                         score_threshold=SCORE_THRESHOLD, fractional_deviation=FRACTIONAL_DEVIATION))

    def testConcatenate2(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 500, 5000))
        network = self.network.copy()
        cnetwork = network.concatenate(self.network)
        for _ in range(10):
            cnetwork = cnetwork.concatenate(self.network)
        self.assertTrue(cnetwork.input_name == "SI")
        self.assertTrue(cnetwork.output_name == "SO")
        # Do simulations
        self.assertTrue(cnetwork.isValid(is_plot=IS_PLOT,
                                         score_threshold=SCORE_THRESHOLD, fractional_deviation=FRACTIONAL_DEVIATION))

    def testBranchjoin(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 100, 1000))
        network = self.network.copy()
        bjn = self.network.branchjoin(network)
        self.assertTrue(bjn.input_name == "SI")
        self.assertTrue(bjn.output_name == "SO")
        # Do simulations
        self.assertTrue(bjn.isValid(is_plot=IS_PLOT,
                                         score_threshold=SCORE_THRESHOLD, fractional_deviation=FRACTIONAL_DEVIATION+0.02))

    def testPfeedback(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 100, 1000))
        fbn = self.network.pfeedback(k5=100)
        self.assertTrue(fbn.input_name == "SI")
        self.assertTrue(fbn.output_name == "SO")
        # Do simulations
        self.assertTrue(fbn.isValid(is_plot=IS_PLOT,
                                         score_threshold=SCORE_THRESHOLD, fractional_deviation=FRACTIONAL_DEVIATION+0.02))
    
    def testNfeedback(self):
        if IGNORE_TEST:
            return
        self.init(model=LINEAR_MDL1, times=np.linspace(0, 100, 1000))
        fbn = self.network.nfeedback(k5=10)
        self.assertTrue(fbn.input_name == "SI")
        self.assertTrue(fbn.output_name == "SO")
        # Do simulations
        _, builder = fbn.plotStaircaseResponse(is_plot=IS_PLOT)
        self.assertTrue(fbn.isValid(is_plot=IS_PLOT,
                                         score_threshold=SCORE_THRESHOLD, fractional_deviation=FRACTIONAL_DEVIATION+0.03))

    def testDebugPfeedback(self):
        if IGNORE_TEST:
            return
        rr = te.loada(FEEDBACK_MDL)
        # SAO->SO
        tf = rr["k3"]/(s + rr["k4"] + rr["k5"])
        ntf = NamedTransferFunction("SAO", "SO", tf)
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        self.assertTrue(score > 0.95)
        # SAI -> SAO
        A_tf = rr["A_kIO"]/(s + rr["A_kO"])
        tf = (s + rr["A_kO"])/(s + rr["A_kO"] + rr["k3"])*A_tf
        ntf = NamedTransferFunction("SAI", "SAO", tf)
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        self.assertTrue(score > 0.95)
        # XI -> SAI
        tf = (rr["k2"])/(s + rr["A_kIO"])
        ntf = NamedTransferFunction("XI", "SAI", tf)
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        self.assertTrue(score > 0.95)
        # SI, S) -> XI
        SI_tf = (rr["k5"])/(s + rr["k2"])
        SO_tf = (rr["k1"])/(s + rr["k2"])
        ntf = NamedTransferFunction(["SI", "SO"], "XI", [SI_tf, SO_tf])
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        self.assertTrue(score > 0.95)
        # XI->SO
        Gp = A_tf*(s + rr["A_kO"])/((s + rr["A_kO"] + rr["k3"])*(s + rr["A_kIO"]))
        tf = rr["k2"]*rr["k3"]*Gp/(s + rr["k4"] + rr["k5"])
        ntf = NamedTransferFunction("XI", "SO", tf)
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        # SI ->SO
        tf = rr["k1"]*rr["k2"]*rr["k3"]*Gp/((s + rr["k4"] + rr["k5"])*(s + rr["k2"]) - rr["k2"]*rr["k3"]*rr["k5"]*Gp)
        ntf = NamedTransferFunction("SI", "SO", tf)
        df, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
       

if __name__ == '__main__':
  unittest.main()