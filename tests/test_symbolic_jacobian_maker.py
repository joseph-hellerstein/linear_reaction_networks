from src.lrn_builder.symbolic_jacobian_maker import SymbolicJacobianMaker

import libsbml # type: ignore
import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict

IGNORE_TEST = False

MODEL = """
J1: -> S1; k1*S1
J2: S2 -> 2 S3 + S1; k2*S2
J3: S2 -> ; k3*S2
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
"""
""" 
self.model = self._makeModel()
self.kinetic_constant_dct: dict = self._makeKineticConstantDct()
self.species_names: list = self._makeSpeciesNames()
self.reaction_description_dct: dict = self._makeReactionDescription()
self.jacobian_smat = self._makeSymbolicJacobian() """


class TestSymbolicJacobianMaker(unittest.TestCase):

    def setUp(self):
        self.maker = SymbolicJacobianMaker(MODEL)
        self.maker.model = self.maker._makeModel()

    def getReactions(self)->Dict[str, libsbml.Reaction]:
        # Gets the reactions from the model
        dct = {}
        for i in range(self.maker.model.getNumReactions()):
            reaction = self.maker.model.getReaction(i)
            reaction_id = reaction.getId()
            dct[reaction_id] = self.maker.model.getReaction(i)
        return dct

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.maker, SymbolicJacobianMaker)

    def testMakeModel(self):
        if IGNORE_TEST:
            return
        self.assertIsNotNone(self.maker.model)
        self.assertEqual(self.maker.model.getNumSpecies(), 3)
        self.assertEqual(self.maker.model.getNumReactions(), 3)

    def testMakeKineticConstantDct(self):
        if IGNORE_TEST:
            return
        kinetic_constant_dct = self.maker._makeKineticConstantDct()
        self.assertIsInstance(kinetic_constant_dct, dict)
        self.assertEqual(len(kinetic_constant_dct), 3)
        self.assertIn("k1", kinetic_constant_dct)
        self.assertIn("k2", kinetic_constant_dct)
        self.assertIn("k3", kinetic_constant_dct)
        trues = [kinetic_constant_dct["k1"] == 1.0,
                kinetic_constant_dct["k2"] == 2.0,
                kinetic_constant_dct["k3"] == 3.0]
        self.assertTrue(all(trues))

    def testMakeSpeciesNames(self):
        if IGNORE_TEST:
            return
        self.maker.model = self.maker._makeModel()
        self.maker.species_names = self.maker._makeSpeciesNames()
        self.assertIsInstance(self.maker.species_names, list)
        self.assertEqual(len(self.maker.species_names), 3)
        trues = [n == self.maker._getSpeciesIndex(f"S{n+1}") for n in range(3)]
        self.assertTrue(all(trues))



if __name__ == '__main__':
    unittest.main()