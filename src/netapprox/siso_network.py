"""
SISO reaction network and operations on it.

Usage:
  net = SISONetwork.makeTwoSpeciesNetwork("S1", "S2", 1, 1)

"""

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

    def __init__(self, input_name:str, output_name:str, model_reference:str, kI:float, kO:float,
                 transfer_function:control.TransferFunction, operating_region=DEFAULT_OPERATION_REGION):
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            kI: Rate at which input is consumed 
            kO: Rate which output is cconsumed
            transfer_function:
        """
        self.input_name = input_name
        self.output_name = output_name
        self.model_reference = model_reference
        self.ctlsb = ctl.ControlSBML(self.model_reference, input_names=[self.input_name], output_names=[self.output_name])
        self.antimony = self.ctlsb.getAntimony()
        self.kI = kI
        self.kO = kO
        self.transfer_function = transfer_function
        self.operating_region = operating_region

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
        raise NotImplementedError("Must implement")
    
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