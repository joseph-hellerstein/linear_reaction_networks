from lrn_builder.network_builder import NetworkBuilder  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore

IGNORE_TEST = False


class TestMakeSequentialAntimony(unittest.TestCase):

    def setUp(self):
        self.num_stage = 3
        self.builder = NetworkBuilder(self.num_stage)

    def testBasicStructure(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony()
        self.assertIsInstance(antimony_str, str)
        self.assertIn("model *sequential_network()", antimony_str)
        self.assertIn("end", antimony_str)

    def testValidAntimony(self):
        if IGNORE_TEST:
            return
        # Test that the generated Antimony can be loaded by Tellurium
        antimony_str = self.builder.makeSequentialAntimony()
        try:
            rr = te.loada(antimony_str)
            self.assertIsNotNone(rr)
        except Exception as e:
            self.fail(f"Generated Antimony is not valid: {e}")

    def testSingleStage(self):
        if IGNORE_TEST:
            return
        builder = NetworkBuilder(1)
        antimony_str = builder.makeSequentialAntimony()
        
        # Should have 2 reactions: S0 -> S1 and S0 -> ;
        self.assertIn("S0 -> S1", antimony_str)
        self.assertIn("S0 -> ;", antimony_str)
        
        # Should have 2 kinetic constants: k1 and k2
        self.assertIn("k1 = 1", antimony_str)
        self.assertIn("k2 = 2", antimony_str)
        
        # Should have 2 species: S0 and S1
        self.assertIn("S0 = 0", antimony_str)
        self.assertIn("S1 = 0", antimony_str)

    def testThreeStages(self):
        if IGNORE_TEST:
            return
        builder = NetworkBuilder(3)
        antimony_str = builder.makeSequentialAntimony()
        
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
        antimony_str = self.builder.makeSequentialAntimony()
        
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
        antimony_str = self.builder.makeSequentialAntimony()
        rr = te.loada(antimony_str)
        
        # Should have num_stage + 1 species (S0, S1, ..., S_num_stage)
        species = rr.getFloatingSpeciesIds()
        self.assertEqual(len(species), self.num_stage + 1)
        
        # Verify species names
        for i in range(self.num_stage + 1):
            expected_species = f"S{i}"
            self.assertIn(expected_species, species)

    def testReactionCount(self):
        if IGNORE_TEST:
            return
        antimony_str = self.builder.makeSequentialAntimony()
        rr = te.loada(antimony_str)
        
        # Should have 2 * num_stage reactions
        # (one forward, one degradation per stage)
        num_reactions = rr.getNumReactions()
        self.assertEqual(num_reactions, 2 * self.num_stage)

    def testInvalidInput(self):
        if IGNORE_TEST:
            return
        # Test that num_stage < 1 raises ValueError
        with self.assertRaises(ValueError):
            _ = NetworkBuilder(0)  # Temporary valid builder
        
        with self.assertRaises(ValueError):
            _ = NetworkBuilder(-1)  # Temporary valid builder

    def testVariousStages(self):
        if IGNORE_TEST:
            return
        # Test that various numbers of stages work correctly
        for num_stage in [1, 2, 5, 10]:
            builder = NetworkBuilder(num_stage)
            antimony_str = builder.makeSequentialAntimony()
            
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
        builder = NetworkBuilder(5)
        antimony_str = builder.makeSequentialAntimony()
        rr = te.loada(antimony_str)
        
        for i in range(1, 11):  # 5 stages = 10 kinetic constants
            value = rr.getValue(f"k{i}")
            self.assertEqual(value, i)

    def testModelStructure(self):
        if IGNORE_TEST:
            return
        # Test that the model has the expected structure
        builder = NetworkBuilder(2)
        antimony_str = builder.makeSequentialAntimony()

        # Split into lines for detailed checking
        lines = [line.strip() for line in antimony_str.split('\n') if line.strip()]
        
        # First line should be model declaration
        self.assertEqual(lines[0], "model *sequential_network()")
        
        # Last line should be end
        self.assertEqual(lines[-1], "end")
        
        # Should contain reaction lines
        reaction_lines = [line for line in lines if '->' in line]
        self.assertEqual(len(reaction_lines), 4)  # 2 stages * 2 reactions each

    def testMakeSymbolicAMatrix(self):
        #if IGNORE_TEST:
        #    return
        A_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])
        A_sym = self.builder.makeSymbolicAMatrix(A_mat)
        # Check the shape
        self.assertEqual(A_sym.shape, (3, 3))
        # Verify that columns sum to 0
        self.assertEqual(0, sum(sp.ones(1, A_sym.rows)*A_sym))
        # Verify that entries are present where expected
        nz_mat = A_mat != 0 + np.eye(A_mat.shape[0])
        nz_sym = A_sym.applyfunc(lambda x: True if x != 0 else False)
        for irow in range(A_mat.shape[0]):
            for icol in range(A_mat.shape[1]):
                self.assertEqual(nz_mat[irow, icol], nz_sym[irow, icol],
                        f"Mismatch at ({irow}, {icol})")



if __name__ == '__main__':
    unittest.main()
