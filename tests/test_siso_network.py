from netapprox.siso_network import SISONetwork # type: ignore

import numpy as np
import pandas as pd
import re
import unittest
import tellurium as te # type: ignore


IGNORE_TEST = False
IS_PLOT = False
LINEAR_MDL = """
model() *main_model
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
class TestSISONetwork(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.init()

    def init(self):
        if "builder" in dir(self):
            return

    def check(self, builder=None):
        if builder is None:
            builder = self.builder
        rr = te.loada(str(builder))
        data = rr.simulate(0,20, 2000, selections=["time", "S1", "S2", "S3"])
        self.assertTrue(len(data) > 0)
        if IS_PLOT:
            rr.plot()
        return data
    
    def testProperties(self):
        if IGNORE_TEST:
            return
        builder = ab.AntimonyBuilder(MTOR_MDL) 
        self.assertGreater(len(builder.floating_species_names), 0)
        self.assertEqual(len(builder.boundary_species_names), 0)
        self.assertGreater(len(builder.reaction_names), 0)
        self.assertGreater(len(builder.parameter_names), 0)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.init()
        self.assertTrue(isinstance(self.builder.antimony, str))

    def getStatement(self, pos=1, builder=None):
        if builder is None:
            builder = self.builder
        return builder.antimony_strs[builder.insert_pos-pos]

    def testMakeComment(self):
        if IGNORE_TEST:
            return
        self.init()
        self.builder.makeComment("comment")
        self.assertTrue("comment" in self.getStatement())

    def testMakeAdditionStatement(self):
        if IGNORE_TEST:
            return
        self.init()
        self.builder.makeAdditionStatement("S1", "S2", "S3")
        result = re.search("S1.*:=.*S2.*\+.*S3", self.getStatement())
        self.assertTrue(result)
        self.builder.makeAdditionStatement("S2", "S3", is_assignment=False)
        result = re.search("S2.* =.*S3", self.getStatement())
        self.assertTrue(result)

    def testCopyAndEqual(self):
        if IGNORE_TEST:
            return
        self.init()
        builder = self.builder.copy()
        self.assertTrue(builder == self.builder)
        #
        builder.makeBoundarySpecies("S1")
        self.assertFalse(builder == self.builder)
       

if __name__ == '__main__':
  unittest.main()