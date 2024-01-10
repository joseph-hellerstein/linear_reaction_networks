"""Representation of a SISO reaction network."""

from netapprox.make_roadrunner import makeRoadrunner

import control  # type: ignore
import controlSBML as ctl # type: ignore
import numpy as np
import pandas as pd
import tellurium as te # type: ignore


DEFAULT_OPERATION_REGION = [0, 10]
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
        roadrunner = makeRoadrunner(model_reference)
        self.antimony = roadrunner.getAntimony()
        self.antimony_strs = [s.strip() for s in self.antimony.split("\n") if len(s.strip()) > 0]
        self.kI = kI
        self.kO = kO
        self.transfer_function = transfer_function
        self.operating_region = operating_region

    def getAntimony(self)->str:
        """
        Returns:
            str: Antimony representation of the network
        """
        return "\n".join(self.antimony_strs)

    # FIXME: Use controlSBML plot 
    def plot(self, times=DEFAULT_TIMES, is_plot=True):
        """
        Args:
            is_plot: If True, plot the transfer function
        """
        rr = te.loada(self.getAntimony())
        rr.simulate(times[0], times[-1], len(times), selections=["time", self.input_name, self.output_name])
        if is_plot:
            rr.plot()