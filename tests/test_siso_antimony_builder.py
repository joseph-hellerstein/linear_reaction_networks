from netapprox import siso_antimony_builder as snb # type: ignore
from netapprox import util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import unittest
import tellurium as te # type: ignore


IGNORE_TEST = True
IS_PLOT = False
MODEL_NAME = "main_model"
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
model *main_model ()
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
"""


#############################
# Tests
#############################
class TestSISONetworkBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = snb.SISOAntimonyBuilder(LINEAR_MDL)

    def check(self, builder=None):
        if builder is None:
            builder = self.builder
        rr = te.loada(str(builder))
        #data = rr.simulate(0,20, 2000, selections=["time", "S1", "S2", "S3"])
        data = rr.simulate(0, 5, 10)
        self.assertTrue(len(data) > 0)
        if IS_PLOT:
            rr.plot()
        return data

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.builder.antimony_strs, list))
        self.check()

    def testMakeComment(self):
        if IGNORE_TEST:
            return
        self.builder.makeComment("comment")
        self.assertTrue("comment" in self.getStatement())

    def testCopyAndEqual(self):
        if IGNORE_TEST:
            return
        builder = self.builder.copy()
        self.assertTrue(builder == self.builder)

    def testAppendModel(self):
        #if IGNORE_TEST:
        #    return
        df1 = util.mat2DF(self.check())
        builder = snb.SISOAntimonyBuilder(TWO_SPECIES_MDL)
        df2 = util.mat2DF(self.check(builder=builder))
        df_full = df1.merge(df2)
        current_len = len(builder.antimony_strs)
        self.builder.appendModel(builder, comment="Simple Model")
        expected_length = len(builder.antimony_strs) + current_len + 8
        actual_length = len(self.builder.antimony_strs) - 3   # Account for end statements
        self.assertEqual(expected_length, actual_length)
        # Show that the same data is produced by the merged models
        df_merged = util.mat2DF(self.check())
        ssq = ((df_merged - df_full)**2).sum().sum()
        self.assertTrue(ssq < 1e-6)
       

if __name__ == '__main__':
  unittest.main()