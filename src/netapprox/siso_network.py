"""
SISO reaction network and operations on it.

Usage:
  net = SISONetwork.makeTwoSpeciesNetwork("S1", "S2", 1, 1)

"""
from netapprox.antimony_template import AntimonyTemplate
from netapprox import constants as cn
from netapprox import util

import control  # type: ignore
import controlSBML as ctl # type: ignore
import numpy as np
import pandas as pd
import tellurium as te # type: ignore
from typing import List, Optional


DEFAULT_OPERATION_REGION = list(np.linspace(0, 10, 5))
DEFAULT_TIMES = list(np.linspace(0, 10, 1000))
MAIN_MODEL_NAME = "main_model"


class SISONetwork(object):
    """
    Representation of a SISO reaction network.
    """

    def __init__(self, antimony_str:str, input_name:str, output_name:str, kI:float, kO:float,
                 transfer_function:control.TransferFunction, operating_region: List[float]=DEFAULT_OPERATION_REGION,
                 children:Optional[List["SISONetwork"]]=None, times:List[float]=DEFAULT_TIMES):
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            antimony_str: Antimony string, possibly with template variables
            kI: rate at which the input is consumed
            kO: rate at which the output is consumed
            transfer_function: transfer function for the network
            children: List of SiSONetworks from which this network is constructed
            times: times at which to simulate the network
        """
        self.input_name = input_name
        self.output_name = output_name
        self.template: AntimonyTemplate = AntimonyTemplate(antimony_str)
        self.kI = kI
        self.kO = kO
        self.operating_region = operating_region
        self.transfer_function = transfer_function
        if children is None:
            children = []
        self.children = children
        self.times = times

    def makeSubmodelName(self, parent_model_name:str, idx:int)->str:
        """
        Args:
            parent_model_name: name of the parent model
            idx: index of the child
        Returns:
            str: name of the child
        """
        return  "%s_%d" % (parent_model_name, idx)

    def getAntimony(self, model_name: Optional[str]=None)->str:
        """
        Recursively expands the model. Replaces template names of the form <child_n> with a name appropriate for the
        expansion. If model_name is None, then it is assumed to be the main model.
        Args:
            model_name: name of the main model
        Returns:
            str
        """
        self.template.initialize()
        def makeNames(idx):
            template_name = self.template.sub_model_name(idx)
            submodel_name = self.makeSubmodelName(model_name, idx)
            return template_name, submodel_name
        #
        if model_name is None:
            model_name = MAIN_MODEL_NAME
        if model_name == MAIN_MODEL_NAME:
            substitute_name = "*%s" % model_name
        else:
            substitute_name = model_name
        self.template.setTemplateVariable(cn.TE_MODEL_NAME, substitute_name)
        # Substitute the template names
        for idx, child in enumerate(self.children):
            template_name, submodel_name = makeNames(idx)
            self.template.setTemplateVariable(template_name, submodel_name)
        antimony_str:str = self.template.substituted_antimony  # type: ignore
        # Recursively replace other antimony
        for child in self.children:
            _, model_name = makeNames(idx)
            antimony_str += "\n" + child.getAntimony(model_name=model_name)
        return antimony_str

    def plotStaircaseResponse(self, initial_value:Optional[float]=None,
                              final_value:Optional[float]=None, num_step:Optional[float]=None,
                              **kwargs):
        """
        Args:
            initial_value: initial value of the input
            final_value: final value of the input
            num_step: number of steps in the staircase
            kwargs: plot options
        """
        if initial_value is None:
            initial_value = self.operating_region[0]
        if final_value is None:
            final_value = self.operating_region[-1]
        if num_step is None:
            num_step = len(self.operating_region)
        ctlsb = ctl.ControlSBML(self.getAntimony(), input_names=[self.input_name], output_names=[self.output_name],
                                is_fixed_input_species=True)
        ctlsb.plotStaircaseResponse(initial_value=initial_value, final_value=final_value, num_step=num_step, **kwargs)

    def simulate(self, **kwargs)->pd.DataFrame:
        """
        Returns:
            pd.DataFrame
                Columns: "time", self.input_name, self.output_name
        """
        rr = te.loada(self.getAntimony())
        for key, value in kwargs.items():
            setattr(rr, key, value)
        data_arr = rr.simulate(self.times[0], self.times[-1], len(self.times),
                               selections=[cn.TIME, self.input_name, self.output_name])
        df = util.mat2DF(data_arr)
        df = df[[cn.TIME, self.input_name, self.output_name]]
        return df

    @classmethod
    def makeTwoSpeciesNetwork(cls, kI:float, kO:float, **kwargs)->"SISONetwork":
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            kI: Rate at which input is consumed 
            kO: Rate which output is cconsumed
            kwargs: additional arguments for constructor
        """
        model = """
        model %s()
        SI -> SO; kIO*SI
        SO -> ; kO*SO
        kIO = %f
        kO = %f
        SI = 0
        SO = 0
        end
        """ % (cn.TE_MODEL_NAME, kI, kO)
        def transfer_function(**kwargs):
            return control.TransferFunction([kwargs["kO"]], [1, kwargs["kI"]])
        return cls(model, "SI", "SO", kI, kO, transfer_function, **kwargs)
    
    @classmethod
    def makeSequentialNetwork(cls, kIOs:List[float], kOs:[float],
                              operating_region=DEFAULT_OPERATION_REGION)->"SISONetwork":
        """
        Creates a sequential network of length len(kIs) = len(kOs). kI = kIs[0]; kO = kOs[-1].
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            kIOs: Rates at which input is consumed 
            kOs: Rates which output is consumed
        """
        if len(kIOs) != len(kOs):
            raise ValueError("kIs and kOs must be the same length")
        model = """
            SI_%d -> S%d; kIO_%d*SI_%d
            SO_%d -> ; kO_%d*SO_%d
            kOI_%d = %f 
            kO_%d = %f

            """
        def makeStage(idx:int)->str:
            return model % (idx, idx, idx, idx, idx, idx, idx, idx, kIOs[idx], idx, kOs[idx])
        antimony_str = "\n".join([makeStage(n) for n in range(len(kIOs))])
        tf1 = control.TransferFunction([kIOs[0]], [1, kIOs[0]])
        tfs = np.prod([control.TransferFunction([kOs[n]], [1, kIOs[n] + kOs[n], kIOs[n]*kOs[n]]) for n in range(2, len(kIOs))])
        transfer_function = tf1*tfs
        return cls("SI", "SO", antimony_str, kIOs[0], kOs[-1], transfer_function)
    
    @classmethod
    def makeCascade(cls, input_name:str, output_name:str, kIs:List[float], kOs:List[float],
                    operating_region=DEFAULT_OPERATION_REGION)->"SISONetwork":
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_references: references in a form that can be read by Tellurium
            kIs: Rates at which input is consumed 
            kOs: Rates which output is cconsumed
        """
        raise NotImplementedError("Must implement")
    
    def sequence(self, other:"SISONetwork")->"SISONetwork":
        """
        Creates a new network that is the concatenation of this network and another.
        Args:
            other: SISONetwork
        Returns:
            SISONetwork
        """
        raise NotImplementedError("Must implement")
    
    def forkjoin(self, other:"SISONetwork")->"SISONetwork":
        """
        Creates a new network by combining this network and another in parallel.
        Args:
            other: SISONetwork
        Returns:
            SISONetwork
        """
        raise NotImplementedError("Must implement")
    
    def loop(self, k1:float, k2:float, k3:float, k4:float, k5:float, k6:float)->"SISONetwork":
        """
        Creates a new network by creating a feedback loop around the existing network. Let N.SI be the input species to the
        current network and N.SO be the output species from the current network. The new network will have the following reactions
        SI -> XI; k1*SI
        XI -> N.SI; k2*XI
        N.SO -> XO; k3*XO
        XO -> SO; k4*XO
        XO -> XI; k5*XO
        SO -> ; k6*SO
        """
        raise NotImplementedError("Must implement")
    
    def amplify(self, k1, k2)->"SISONetwork":
        """
        Creates a new network by amplifying the output of the current network. Let N.SI be the input species to the
        current network and N.SO be the output species from the current network. The new network will have the following reactions
        SI -> N.S; k1*SI
        N.S -> SO; k2*N.S
        """
        raise NotImplementedError("Must implement")