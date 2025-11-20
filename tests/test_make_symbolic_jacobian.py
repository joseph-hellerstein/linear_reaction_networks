from src.lrn_builder.make_symbolic_jacobian import makeSymbolicJacobian

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False

MODEL = """
S1 -> S2; k1*S1
S2 -> 2 S3; k2*S2
S2 -> ; k3*S2
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
"""

def isSymbol(expression, symbol_name):
    return expression.free_symbols.pop().name == symbol_name


class TestMakeSymbolicJacobian(unittest.TestCase):

    def setUp(self):
        pass

    def testBasicJacobian(self):
        if IGNORE_TEST:
            return
        # Test that the generated Antimony can be loaded by Tellurium
        antimony_str = MODEL
        try:
            result = makeSymbolicJacobian(antimony_str)
            jacobian_smat = result.jacobian_smat
            kinetic_constant_dct = result.kinetic_constant_dct
            self.assertIsInstance(jacobian_smat, sp.Matrix)
            self.assertIsInstance(kinetic_constant_dct, dict)
            self.assertTrue(isSymbol(jacobian_smat[0,0], 'k1'))
            self.assertTrue(isSymbol(jacobian_smat[1,0], 'k1'))
            self.assertTrue(isSymbol(jacobian_smat[2,1], 'k2'))
        except Exception as e:
            self.fail(f"makeSymbolicJacobian failed: {e}") 


if __name__ == '__main__':
    unittest.main()