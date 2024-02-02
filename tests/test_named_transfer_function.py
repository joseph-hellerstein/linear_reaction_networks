from lrn_builder.named_transfer_function import NamedTransferFunction # type: ignore

import control # type: ignore
import numpy as np
import unittest
import pandas as pd


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
TRANFER_FUNCTION = control.TransferFunction([1], [1, 2])
TIMES = list(np.linspace(0, 10, 100))


#############################
# Tests
#############################
class TestNamedTransferFunction(unittest.TestCase):

    def setUp(self):
        self.ntf = NamedTransferFunction("S1", "S2", TRANFER_FUNCTION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.ntf.input_name, "S1")
        self.assertTrue(isinstance(self.ntf.transfer_function, control.TransferFunction))
    
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
        diff = set(["time", "simulation", "input"]) - set(df.columns)
        self.assertEqual(len(diff), 0)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def testPredict(self):
        if IGNORE_TEST:
            return
        df = self.ntf.predict(LINEAR_MDL, TIMES)
        self.assertTrue(isinstance(df, pd.DataFrame))
        diff = set(["time", "simulation", "input", "prediction"]) - set(df.columns)
        self.assertEqual(len(diff), 0)

    def testVerify(self):
        if IGNORE_TEST:
            return
        score = self.ntf.score(LINEAR_MDL, TIMES, is_plot=IS_PLOT)
        self.assertGreater(score, 0.95)

       

if __name__ == '__main__':
  unittest.main()