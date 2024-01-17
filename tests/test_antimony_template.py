from netapprox.antimony_template import AntimonyTemplate  # type: ignore
from netapprox import util # type: ignore
from netapprox import constants as cn # type: ignore

import unittest
import tellurium as te # type: ignore


IGNORE_TEST = False
IS_PLOT = False
MODEL_NAME = "a_model"
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
model *%s ()
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
""" % MODEL_NAME
BARE_MDL = """
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
"""


#############################
# Tests
#############################
class TestSISONetworkBuilder(unittest.TestCase):

    def setUp(self):
        self.template = AntimonyTemplate(LINEAR_MDL)

    def check(self, template=None):
        if template is None:
            template = self.template
        rr = te.loada(str(template))
        #data = rr.simulate(0,20, 2000, selections=["time", "S1", "S2", "S3"])
        data = rr.simulate(0, 5, 10)
        self.assertTrue(len(data) > 0)
        if IS_PLOT:
            rr.plot()
        return data

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.template.substituted_antimony, str))
        self.assertEqual(self.template.model_name, MODEL_NAME)
        #
        template = AntimonyTemplate(BARE_MDL)
        template.setTemplateVariable(cn.TE_MODEL_NAME, MODEL_NAME)
        self.assertTrue(template.isValidAntimony())

    def testCopyAndEqual(self):
        if IGNORE_TEST:
            return
        template = self.template.copy()
        self.assertTrue(template == self.template)

    def testSetTemplateVariable(self):
        if IGNORE_TEST:
            return
        template = self.template.copy()
        template.setTemplateVariable(cn.TE_MODEL_NAME, "*b_model")
        self.assertTrue(template.substituted_antimony.count("*b_model") == 1)
        self.assertTrue(template != self.template)
        self.check(template)

    def testIsValidAntimony(self):
        if IGNORE_TEST:
            return
        self.assertFalse(self.template.isValidAntimony())
        self.template.setTemplateVariable(cn.TE_MODEL_NAME, "*b_model")
        self.assertTrue(self.template.isValidAntimony())

    def testMakeSubmodelTemplateName(self):
        if IGNORE_TEST:
            return
        submodel_name = self.template.makeSubmodelTemplateName(1)
        self.assertEqual(submodel_name, "<<submodel_name__1>>")

       

if __name__ == '__main__':
  unittest.main()