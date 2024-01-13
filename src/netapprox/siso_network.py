"""
SISO reaction network and operations on it.

Usage:
  net = SISONetwork.makeTwoSpeciesNetwork("S1", "S2", 1, 1)

"""
from netapprox.siso_antimony import SISOAntimony

import control  # type: ignore
import controlSBML as ctl # type: ignore
import numpy as np
import pandas as pd
import tellurium as te # type: ignore
from typing import List, Optional


DEFAULT_OPERATION_REGION = np.linspace(0, 10, 5)
DEFAULT_TIMES = np.linspace(0, 10, 1000)


class SISONetwork(object):
    """
    Representation of a SISO reaction network.
    """

    def __init__(self, input_name:str, output_name:str, model_reference:str, kI:str, kO:str,
                 transfer_function_generator:function, operating_region=DEFAULT_OPERATION_REGION,
                 transfer_function_parameters:Optional[List[str]]=None,
                 children:Optional[List["SISONetwork"]]=None):
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            kI: parameter in roadrunner model for rate at which input is consumed
            kO: parameter in roadrunner model for rate at which output is consumed
            transfer_function_generator: Function(**kwargs) -> control.TransferFunction
            children: List of SiSONetworks from which this network is constructed
        """
        self.input_name = input_name
        self.output_name = output_name
        self.model_reference = model_reference
        self.ctlsb = ctl.ControlSBML(self.model_reference, input_names=[self.input_name], output_names=[self.output_name])
        self.antimony = SISOAntimony(self.ctlsb.getAntimony())
        self.kI = kI
        self.kO = kO
        self.transfer_function_generator = transfer_function_generator
        if transfer_function_parameters is None:
            transfer_function_parameters = []
        self.transfer_function_parameters = transfer_function_parameters
        self.operating_region = operating_region
        if children is None:
            children = []
        self.children = children

    def getAntimony(self, model_name: Optional[str]=None, is_main:bool=False)->str:
        """
        Recursively expands the mondel. Replaces template names of the form <child_n> with a name appropriate for the
        expansion.
        Args:
            model_name: name of the model to return
            is_main: if True, returns the main model
        Returns:
            str
        """
        def makeChildNames(idx):
            old_name = "<child_%d>" % idx
            new_name = "%s_%d" % (model_name, idx)
            return old_name, new_name
        #
        if model_name is None:
            model_name = self.antimony.main_model_name
        antimony_str = self.getAntimony(model_name=model_name, is_main=is_main)
        # Substitute the template names
        for idx, child in enumerate(self.children):
            old_name, new_name = makeChildNames(idx)
            antimony_str = antimony_str.replace(old_name, new_name)
        # Recursively replace other antimony
        for child in self.children:
            _, new_name = makeChildNames(idx)
            antimony_str += child.getAntimony(model_name=new_name, is_main=False)
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
        self.ctlsb.plotStaircaseResponse(initial_value=initial_value, final_value=final_value, num_step=num_step, **kwargs)

    @classmethod
    def makeTwoSpeciesNetwork(cls, kI:float, kO:float,
                              operating_region=DEFAULT_OPERATION_REGION)->"SISONetwork":
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            kI: Rate at which input is consumed 
            kO: Rate which output is cconsumed
        """
        model = """
        model main_model()
        SI -> SO; kIO*SI
        SO -> ; kO*SO
        kI = 1
        kO = 1
        end
        """
        def transfer_function(**kwargs):
            return control.TransferFunction([kwargs["kO"]], [1, kwargs["kI"]])
        return cls("SI", "SO", model, "kI", "kO", transfer_function)
    
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
    
    def concatenate(self, other:"SISONetwork")->"SISONetwork":
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