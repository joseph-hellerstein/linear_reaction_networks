from lrn_builder.make_lti_crn import makeLtiCrn  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False



class TestGenerateCrn(unittest.TestCase):

    def setUp(self):
        pass

    def testBasicJacobian(self):
        if IGNORE_TEST:
            return
        for _ in range(5):
            max_num_reaction = np.random.randint(1, 10)
            max_num_product = np.random.randint(1, 10)
            max_kinetic_constant = np.random.uniform(1, 10)
            max_stoichiometry = np.random.randint(1, 10)
            try:
                model = makeLtiCrn(np.random.randint(1, 10),
                    np.random.randint(1, max_num_reaction),
                    num_products_bounds=(1, max_num_product),
                    kinetic_constant_bounds=(0, max_kinetic_constant),
                    stoichiometry_bounds=(1, max_stoichiometry),
                    seed=42  # For reproducibility
                )
            except:
                self.fail("generateCrn raised an exception unexpectedly!")
            try:
                rr = te.loada(model)  # type: ignore
                self.assertIsNotNone(rr)
            except Exception as e:
                self.fail(f"Generated Antimony is not valid: {e}")
            try:
                data = rr.simulate(0, 10, 100)
                self.assertTrue(data.shape[0] > 0)
            except Exception as e:
                self.fail(f"Could not simulate the generated model: {e}")


if __name__ == '__main__':
    unittest.main()