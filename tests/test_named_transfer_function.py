from lrn_builder.named_transfer_function import NamedTransferFunction # type: ignore

import control # type: ignore
import numpy as np
import unittest
import pandas as pd
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
TRANFER_FUNCTION = control.TransferFunction([1], [1, 2])
TIMES = list(np.linspace(0, 10, 100))
s = control.TransferFunction.s

# TODO: Add tests for MISO
#############################
# Tests
#############################
class TestNamedTransferFunction(unittest.TestCase):

    def setUp(self):
        self.ntf = NamedTransferFunction("S1", "S2", TRANFER_FUNCTION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.ntf.input_names, ["S1"])
        self.assertTrue(isinstance(self.ntf.transfer_functions[0], control.TransferFunction))
    
    def testRepr(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(repr(self.ntf), str))

    def testEq(self):
        if IGNORE_TEST:
            return
        ntf = self.ntf.copy()
        self.assertTrue(self.ntf == ntf)
        transfer_function = control.TransferFunction([1], [1, 3])
        ntf = NamedTransferFunction("S1", "S2", transfer_function)
        self.assertFalse(self.ntf == ntf)

    def testSimulate(self):
        if IGNORE_TEST:
            return
        df = self.ntf.simulate(LINEAR_MDL, TIMES)
        diff = set(["time", "simulation", "input__S1"]) - set(df.columns)
        self.assertEqual(len(diff), 0)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def testPredict(self):
        if IGNORE_TEST:
            return
        df = self.ntf.predict(LINEAR_MDL, TIMES)
        self.assertTrue(isinstance(df, pd.DataFrame))
        diff = set(["time", "simulation", "input__S1", "prediction"]) - set(df.columns)
        self.assertEqual(len(diff), 0)

    def testVerify(self):
        if IGNORE_TEST:
            return
        score = self.ntf.score(LINEAR_MDL, TIMES, is_plot=IS_PLOT)
        self.assertGreater(score, 0.95)

    def testMISOEvaluate(self):
        if IGNORE_TEST:
            return
        rr = te.loada(FEEDBACK_MDL)
        SI_tf = (rr["k5"])/(s + rr["k2"])
        SO_tf = (rr["k1"])/(s + rr["k2"])
        ntf = NamedTransferFunction(["SI", "SO"], "XI", [SI_tf, SO_tf])
        _, score = ntf.evaluate(FEEDBACK_MDL, is_plot=IS_PLOT)
        self.assertGreater(score, 0.95)
       

if __name__ == '__main__':
  unittest.main()