from lrn_builder.siso_analyzer import SISOAnalyzer  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
from typing import Optional

IGNORE_TEST = False

MODEL = """
S1_ -> S2_; k201*S1_
S2 -> S3; k302*S2
S1_ = 10
S2 = 0
S3 = 0
k201 = 1
k302 = 2
"""


class TestMakeSequentialAntimony(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.num_stage = 3
        self.builder = SISOAnalyzer(MODEL)

    def testBasicStructure(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        self.assertIsInstance(antimony_str, str)
        self.assertIn("model *sequential_network()", antimony_str)
        self.assertIn("end", antimony_str)

    def testValidAntimony(self):
        if IGNORE_TEST:
            return
        # Test that the generated Antimony can be loaded by Tellurium
        antimony_str = self.builder.makeSequentialAntimony(3)
        try:
            rr = te.loada(antimony_str)
            self.assertIsNotNone(rr)
        except Exception as e:
            self.fail(f"Generated Antimony is not valid: {e}")

    def testSingleStage(self):
        if IGNORE_TEST:
            return
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(1)
        
        # Should have 2 reactions: S0 -> S1_ and S0 -> ;
        self.assertIn("S0 -> S1", antimony_str)
        self.assertIn("S0 -> ;", antimony_str)
        
        # Should have 2 kinetic constants: k1 and k2
        self.assertIn("k1 = 1", antimony_str)
        self.assertIn("k2 = 2", antimony_str)
        
        # Should have 2 species: S0 and S1_
        self.assertIn("S0 = 0", antimony_str)
        self.assertIn("S1 = 0", antimony_str)

    def testThreeStages(self):
        if IGNORE_TEST:
            return
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(3)
        
        # Check reactions
        self.assertIn("S0 -> S1; k1*S0", antimony_str)
        self.assertIn("S0 -> ; k2*S0", antimony_str)
        self.assertIn("S1 -> S2; k3*S1", antimony_str)
        self.assertIn("S1 -> ; k4*S1", antimony_str)
        self.assertIn("S2 -> S3; k5*S2", antimony_str)
        self.assertIn("S2 -> ; k6*S2", antimony_str)
        
        # Check kinetic constants
        for i in range(1, 7):
            self.assertIn(f"k{i} = {i}", antimony_str)
        
        # Check species initialization
        for i in range(4):
            self.assertIn(f"S{i} = 0", antimony_str)

    def testKineticConstants(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        
        # Load model and verify kinetic constant values
        rr = te.loada(antimony_str)
        
        # For 3 stages, we should have 6 kinetic constants (k1 through k6)
        expected_num_constants = 2 * self.num_stage
        params = rr.getGlobalParameterIds()
        self.assertEqual(len(params), expected_num_constants)
        
        # Verify each constant has the correct value
        for i in range(1, expected_num_constants + 1):
            param_name = f"k{i}"
            self.assertIn(param_name, params)
            value = rr.getValue(param_name)
            self.assertEqual(value, i, f"Expected {param_name} = {i}, got {value}")

    def testSpeciesCount(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        rr = te.loada(antimony_str)
        
        # Should have num_stage + 1 species (S0, S1_, ..., S_num_stage)
        species = rr.getFloatingSpeciesIds()
        self.assertEqual(len(species), self.num_stage + 1)
        
        # Verify species names
        for i in range(self.num_stage + 1):
            expected_species = f"S{i}"
            self.assertIn(expected_species, species)

    def testReactionCount(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony(3)
        rr = te.loada(antimony_str)
        
        # Should have 2 * num_stage reactions
        # (one forward, one degradation per stage)
        num_reactions = rr.getNumReactions()
        self.assertEqual(num_reactions, 2 * self.num_stage)

    def testVariousStages(self):
        if IGNORE_TEST:
            return
        # Test that various numbers of stages work correctly
        for num_stage in [1, 2, 5, 10]:
            builder = SISOAnalyzer(MODEL)
            antimony_str = builder.makeSequentialAntimony(num_stage)
            
            # Verify it's valid
            rr = te.loada(antimony_str)
            
            # Verify species count
            species = rr.getFloatingSpeciesIds()
            self.assertEqual(len(species), num_stage + 1)
            
            # Verify parameter count
            params = rr.getGlobalParameterIds()
            self.assertEqual(len(params), 2 * num_stage)
            
            # Verify reaction count
            self.assertEqual(rr.getNumReactions(), 2 * num_stage)

    def testSequentialKineticValues(self):
        if IGNORE_TEST:
            return
        # Ensure kinetic constants are truly sequential (1, 2, 3, ...)
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(5)
        rr = te.loada(antimony_str)
        
        for i in range(1, 11):  # 5 stages = 10 kinetic constants
            value = rr.getValue(f"k{i}")
            self.assertEqual(value, i)

    def testModelStructure(self):
        if IGNORE_TEST:
            return
        # Test that the model has the expected structure
        builder = SISOAnalyzer(MODEL)
        antimony_str = builder.makeSequentialAntimony(2)

        # Split into lines for detailed checking
        lines = [line.strip() for line in antimony_str.split('\n') if line.strip()]
        
        # First line should be model declaration
        self.assertEqual(lines[0], "model *sequential_network()")
        
        # Last line should be end
        self.assertEqual(lines[-1], "end")
        
        # Should contain reaction lines
        reaction_lines = [line for line in lines if '->' in line]
        self.assertEqual(len(reaction_lines), 4)  # 2 stages * 2 reactions each

    def testBug1(self):
        if IGNORE_TEST:
            return
        model = """
            -> S1_; k1
            S7_ -> 3 S3_ + S5_ + S3_ + 2 S8_; k2 * S7_
            S3_ -> S5_ + 3 S4_ + S8_ + 3 S9_; k3 * S3_
            S6_ -> S6_ + S7_ + S7_ + S7_ + 3 S3_; k4 * S6_
            S7_ -> S8_ + 2 S10_ + S6_ + S9_ + 3 S3_; k5 * S7_
            S1_ -> S9_ + 3 S9_ + 3 S8_ + 3 S7_ + S9_; k6 * S1_
            S7_ -> 2 S3_ + S5_; k7 * S7_
            S2_ -> S8_ + S5_ + 2 S7_ + 2 S10_; k8 * S2_
            S10_ -> S10_ + 2 S10_ + 3 S7_ + S4_; k9 * S10_
            S3_ -> S3_ + 2 S10_ + S8_; k10 * S3_

            # Rate constants
            k1 = 0.1668
            k2 = 0.9288
            k3 = 0.6378
            k4 = 0.4359
            k5 = 0.3351
            k6 = 0.6957
            k7 = 0.3313
            k8 = 0.5887
            k9 = 0.2808
            k10 = 0.4394

            # Species initialization
            S1_ = 1  # Input boundary species
            S2_ = 0
            S3_ = 0
            S4_ = 0
            S5_ = 0
            S6_ = 0
            S7_ = 0
            S8_ = 0
            S9_ = 0
            S10_ = 0


            # Degradation reactions
            S7_ -> ; kd_S7_ * S7_
            kd_S7_ = 1.9055
            S3_ -> ; kd_S3_ * S3_
            kd_S3_ = 6.6583
            S5_ -> ; kd_S5_ * S5_
            kd_S5_ = 2.7353
            S8_ -> ; kd_S8_ * S8_
            kd_S8_ = 4.2445
            S4_ -> ; kd_S4_ * S4_
            kd_S4_ = 2.4136
            S9_ -> ; kd_S9_ * S9_
            kd_S9_ = 2.4734
            S6_ -> ; kd_S6_ * S6_
            kd_S6_ = 0.3686
            S10_ -> ; kd_S10_ * S10_
            kd_S10_ = 3.6168
        """
        analyzer = SISOAnalyzer(model)

    def testBug2(self):
        if IGNORE_TEST:
            return
        model = """
        model random_crn()

        -> $S1_; k1
        S5_ -> 2 S4_ + 3 S4_ + S4_ + 2 S4_ + 3 S4_; k2 * S5_
        S1_ -> 3 S4_ + 2 S5_ + 3 S5_ + S3_ + S3_; k3 * S1_
        S2_ -> 3 S4_ + S3_ + 2 S3_ + S4_ + 3 S5_; k4 * S2_
        S1_ -> S5_; k5 * S1_
        S2_ -> 3 S3_ + 2 S3_ + 2 S3_ + S5_ + S5_; k6 * S2_
        S1_ -> S4_ + 2 S5_ + 2 S4_ + S3_ + S5_; k7 * S1_
        S2_ -> S4_ + 2 S7_ + S3_ + 3 S6_; k8 * S2_
        S6_ -> ; k9 * S6_
        S7_ -> ; k9 * S7_
        S2_ -> S4_ + 3 S3_; k9 * S2_
        S1_ -> S5_ + 2 S4_ + S4_; k10 * S1_

        # Rate constants
        k1 = 0.9742
        k2 = 0.7012
        k3 = 0.1698
        k4 = 0.4629
        k5 = 0.5307
        k6 = 0.4652
        k7 = 0.8688
        k8 = 0.8631
        k9 = 0.7551
        k10 = 0.1057

        # Species initialization
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        S5_ -> ; kd_0 * S5_
        kd_0 = 3.6785
        S4_ -> ; kd_1 * S4_
        kd_1 = 12.3013
        S3_ -> ; kd_2 * S3_
        kd_2 = 11.3991
        end
        """
        analyzer = SISOAnalyzer(model, output_name="S5_")
        # Make sure that the Jacobian is computed correctly
        jacobian_df = analyzer.jacobian_df
        self.assertIsNotNone(jacobian_df)
        self.assertEqual(jacobian_df.shape[0], jacobian_df.shape[1]) 
        #tf1 = analyzer.transfer_function_expr
        tf1 = analyzer.makeTransferFunction()
        self.assertGreater(tf1.num[-1]/tf1.den[-1], 0) # type: ignore



if __name__ == '__main__':
    unittest.main()
