from lrn_builder.named_transfer_function import NamedTransferFunction # type: ignore

import control # type: ignore
import controlSBML as ctl # type: ignore
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
TIMES = list(np.linspace(0, 10, 100))


#############################
# Tests
#############################
class TestNamedTransferFunction(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.init()

    def init(self, model=LINEAR_MDL, times=TIMES):
        k1 = 1
        k2 = 2
        tf = control.TransferFunction([k1], [1, k2])
        self.network = SLMNetwork(model, "S1", "S2", k1, k2, tf, times=times)

       

if __name__ == '__main__':
  unittest.main()