"""
SISO linear Modular (SLM) reaction network. Describes the network as both an Antimony model and a transfer function.

Usage:
  net = SLMNetwork.makeTwoSpeciesNetwork("S1", "S2", 1, 1)

"""
from lrn_builder.antimony_template import AntimonyTemplate
from lrn_builder.named_transfer_function import NamedTransferFunction
from lrn_builder import constants as cn

import control  # type: ignore
import controlSBML as ctl # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tellurium as te # type: ignore
from typing import List, Optional, Union


DEFAULT_OPERATION_REGION = list(np.linspace(0, 10, 5))
DEFAULT_TIMES = list(np.linspace(0, 10, 1000))
MAIN_MODEL_NAME = "main_model"
PREDICTION = "prediction"
SIMULATION = "simulation"
# Plots
FIGSIZE = "figsize"
IS_PLOT = "is_plot"


class SLMNetwork(object):
    """
    Representation of a SISO reaction network.
    """

    def __init__(self, antimony_str:str, input_name:str, output_name:str, kI:float, kO:float,
                 transfer_function:Union[control.TransferFunction, NamedTransferFunction],
                 operating_region: List[float]=DEFAULT_OPERATION_REGION,
                 children:Optional[List["SLMNetwork"]]=None, times:List[float]=DEFAULT_TIMES):
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            antimony_str: Antimony string, possibly with template variables
            operating_region: Acceptable range of input values
            kI: rate at which the input is consumed
            kO: rate at which the output is consumed
            transfer_function: transfer function for the network
            children: List of SiSONetworks from which this network is constructed
            times: times at which to simulate the network
        """
        self.template: AntimonyTemplate = AntimonyTemplate(antimony_str)
        self.input_name = input_name
        self.output_name = output_name
        self.kI = kI
        self.kO = kO
        # FIXME: NamedTransferFunction
        if isinstance(transfer_function, control.TransferFunction):
            transfer_function = NamedTransferFunction(input_name, output_name, transfer_function)
        self.named_transfer_function = transfer_function
        self.operating_region = operating_region
        if children is None:
            children = []
        self.children = children
        self.times = times

    def __eq__(self, other)->bool:
        """
        Returns:
            bool
        """
        IS_DEBUG = False
        is_true = True
        if not isinstance(other, SLMNetwork):
            if not is_true and IS_DEBUG:
               print("**Failed 0") 
            return False
        is_true = self.template == other.template
        if not is_true and IS_DEBUG:
            print("**Failed 1") 
        is_true = is_true and (self.input_name == other.input_name)
        if not is_true and IS_DEBUG:
            print("**Failed 2") 
        is_true = is_true and (self.output_name == other.output_name)
        if not is_true and IS_DEBUG:
            print("**Failed 3") 
        is_true = is_true and (self.kI == other.kI)
        if not is_true and IS_DEBUG:
            print("**Failed 4") 
        is_true = is_true and (self.kO == other.kO)
        if not is_true and IS_DEBUG:
            print("**Failed 5") 
        is_true = is_true and (self.named_transfer_function == other.named_transfer_function)
        if not is_true and IS_DEBUG:
            print("**Failed 6") 
        is_true = is_true and (self.operating_region == other.operating_region)
        if not is_true and IS_DEBUG:
            print("**Failed 7") 
        is_true = is_true and (self.children == other.children)
        if not is_true and IS_DEBUG:
            print("**Failed 8") 
        is_true = is_true and (np.allclose(self.times, other.times))
        if not is_true and IS_DEBUG:
            print("**Failed 9")              
        return is_true

    def copy(self)->"SLMNetwork":
        """
        Returns:
            SLMNetwork
        """
        network = SLMNetwork(self.template.original_antimony, self.input_name, self.output_name, self.kI, self.kO,  # type: ignore
                           self.named_transfer_function, self.operating_region, self.children, self.times)
        network.template = self.template.copy()
        return network

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
            template_name = self.template.makeSubmodelTemplateName(idx)
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
            template_name, submodel_name = makeNames(idx+1)
            self.template.setTemplateVariable(template_name, submodel_name)
        antimony_str:str = self.template.substituted_antimony  # type: ignore
        # Recursively replace other antimony
        for idx, child in enumerate(self.children):
            _, submodel_name = makeNames(idx+1)
            antimony_str = child.getAntimony(model_name=submodel_name) + "\n" + antimony_str
        return antimony_str

    def plotStaircaseResponse(self, initial_value:Optional[float]=None,
                              final_value:Optional[float]=None, num_step:Optional[float]=None,
                              **kwargs)->tuple[ctl.Timeseries, ctl.AntimonyBuilder]:
        """
        Args:
            initial_value: initial value of the input
            final_value: final value of the input
            num_step: number of steps in the staircase
            kwargs: plot options
        Returns:
            tuple[ctl.Timeseries, ctl.AntimonyBuilder]
            
        """
        if initial_value is None:
            initial_value = self.operating_region[0]
        if final_value is None:
            final_value = self.operating_region[-1]
        if num_step is None:
            num_step = len(self.operating_region)
        ctlsb = ctl.ControlSBML(self.getAntimony(), input_names=[self.input_name], output_names=[self.output_name],
                                is_fixed_input_species=True)
        result = ctlsb.plotStaircaseResponse(initial_value=initial_value, final_value=final_value, num_step=num_step, **kwargs)
        return result
    
    def isValid(self, score_threshold:float=0.95, **kwargs)->bool:
        """
        Compares the transfer function output to the simulation output for a staircase input.
        Args:
            score_threshold: threshold for the fraction of absolute errors < 0.01
            kwargs: Staircase options
        Returns:
            bool: True if the network is valid
        """
        antimony_model = str(self.getStaircaseAntimony(**kwargs))
        return self.named_transfer_function.score(antimony_model, **kwargs) >= score_threshold
    
    def getStaircaseAntimony(self, **kwargs)->str:
        """
        Provides the antimony used to construct a staircase response.
        Args:
            kwargs: Staircase options
        Returns:
            str (Antimony model)
        """
        new_kwargs = dict(kwargs)
        new_kwargs[IS_PLOT] = False
        if not "times" in new_kwargs:
            new_kwargs["times"] = self.times
        _, builder = self.plotStaircaseResponse(**new_kwargs)
        return builder

    
    ################# NETWORK CONSTRUCTION ###############
    @classmethod
    def makeTwoSpeciesNetwork(cls, kI:float, kO:float, **kwargs)->"SLMNetwork":
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
        model *%s()
        SI -> SO; kIO*SI
        SO -> ; kO*SO
        kIO = %f
        kO = %f
        SI = 0
        SO = 0
        end
        """ % (MAIN_MODEL_NAME, kI, kO)
        transfer_function = control.TransferFunction([kI], [1, kO])
        return cls(model, "SI", "SO", kI, kO, transfer_function, **kwargs)
    
    @classmethod
    def makeSequentialNetwork(cls, ks:List[float], kps:[float],
                              operating_region=DEFAULT_OPERATION_REGION)->"SLMNetwork":
        """
        Creates a sequential network of length len(kIs) = len(kOs). kI = kIs[0]; kO = kOs[-1].
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_reference: reference in a form that can be read by Tellurium
            ks: Rate at whcih Species n produces Species n+1 (n < N)
            kps: Rates which Species n is degraded (n > 0)
        """
        if len(ks) != len(kps):
            raise ValueError("kIs and kOs must be the same length")
        model = """
            S%d -> S%d; k_%d*S%d
            S%d -> ; kp_%d*S%d
            k_%d = %f 
            kp_%d = %f
            S%d = 0
            """
        def makeStage(idx:int)->str:
            return model % (idx-1, idx, idx-1, idx-1, idx, idx, idx, idx-1, ks[idx-1], idx, kps[idx-1], idx)
        stages = [makeStage(n) for n in range(1, len(ks)+1)]
        antimony_str = "\n".join(stages)
        model_str = """
            model *%s()
            %s
            S0 = 0
            end
        """ % (MAIN_MODEL_NAME, antimony_str)
        tf1 = control.TransferFunction([ks[0]], [1, kps[-1]])
        new_ks = ks[1:]
        new_kps = kps[:-1]
        tfn = np.prod([control.TransferFunction([new_ks[n]], [1, new_ks[n] + new_kps[n]])
                                      for n in range(len(new_ks))])
        transfer_function = tf1*tfn
        output_name = "S%d" % len(ks)
        kI = ks[0]
        kO = kps[-1]
        return cls(model_str, "S0", output_name, kI, kO, transfer_function)
    
    @classmethod
    def makeCascade(cls, input_name:str, output_name:str, kIs:List[float], kOs:List[float],
                    operating_region=DEFAULT_OPERATION_REGION)->"SLMNetwork":
        """
        Args:
            input_name: input species to the network
            output_name: output species from the network
            model_references: references in a form that can be read by Tellurium
            kIs: Rates at which input is consumed 
            kOs: Rates which output is cconsumed
        """
        raise NotImplementedError("Must implement")

    ################# NETWORK OPERATIONS ###############
    def concatenate(self, other:"SLMNetwork")->"SLMNetwork":
        """
        Creates a new network that is the concatenation of this network and another.
        Args:
            other: SLMNetwork
        Returns:
            SLMNetwork
        """
        submodel1 = self.template.makeSubmodelTemplateName(1)
        submodel2 = self.template.makeSubmodelTemplateName(2)
        model = """
            A: %s();
            B: %s();
            A.%s is B.%s;
            SI is A.%s
            SO is B.%s
            """ % (submodel1, submodel2, self.output_name, other.input_name, self.input_name, other.output_name)
        self_tf = self.named_transfer_function.transfer_function
        other_tf = other.named_transfer_function.transfer_function
        transfer_function = self_tf*other_tf*control.TransferFunction(
            [1, self.kO], [1, self.kO + other.kI])
        network = SLMNetwork(model, "SI", "SO", self.kI, other.kO, transfer_function,
                              operating_region=self.operating_region, times=self.times,
                              children=[self, other])
        return network

    # FIXME: Do I have the correct operating region? 
    def branchjoin(self, other:"SLMNetwork", k1a:float=1, k1b:float=1, k2a:float=1, k2b:float=1, k3:float=1)->"SLMNetwork":
        """
        Creates a new network by combining this network and another in parallel.
        Args:
            other: SLMNetwork
            k1a: (float) SI->Module A
            k1b: (float) SI->Module B
            k2a: (float) Module A -> SO
            k2b: (float) Module B -> SO
            k3: (float) SO -> emptyset
        Returns:
            SLMNetwork
        """
        submodel1 = self.template.makeSubmodelTemplateName(1)
        submodel2 = self.template.makeSubmodelTemplateName(2)
        model = """
            A: %s();
            B: %s();
            species SI, SO
            SAI is A.%s
            SBI is B.%s
            SAO is A.%s
            SBO is B.%s
            SI -> SAI; k1a*SI
            SI -> SBI; k1b*SI
            SAO -> SO; k2a*SAO
            SBO -> SO; k2b*SBO
            SO -> ; k3*SO
            k1a = %f
            k1b = %f
            k2a = %f
            k2b = %f
            k3 = %f
            """ % (submodel1,  submodel2, self.input_name, other.input_name,
                   self.output_name, other.output_name, k1a, k1b, k2a, k2b, k3)
        # Calculate the transfer function of the composition
        tf = self.named_transfer_function.transfer_function
        other_tf = other.named_transfer_function.transfer_function
        s = control.TransferFunction.s
        alpha_tf = (s + self.kI)*(s + self.kO + k2a)
        beta_tf = (s + other.kI)*(s + other.kO + k2b)
        numr1 = k1a*k2a*(s + self.kO)*beta_tf*tf
        denom = alpha_tf*beta_tf*(s + k3)
        numr2 = k1b*k2b*(s + other.kO)*alpha_tf*other_tf
        transfer_function = (numr1 + numr2) / denom
        kI = self.kI + other.kI
        kO = k3
        operating_region = list(np.array(self.operating_region) + np.array(other.operating_region))
        network = SLMNetwork(model, "SI", "SO", kI, kO, transfer_function,
                              operating_region=operating_region, times=self.times,
                              children=[self, other])
        return network
    
    def pfeedback(self, k1:float=1, k2:float=1, k3:float=1, k4:float=1, k5:float=1)->"SLMNetwork":
        """
        Creates a new network by creating a positive feedback loop around the existing network.

        Args:
            k1: (float) kinetic constant for converting SI into XI
            k2: (float) kinetic constant for converting XI into self.SI
            k3: (float) kinetic constant for converting self.SO into SO
            k4: (float) kinetic constant for degrading SO
            k5: (float) kinetic constant for converting self.SO into XI
        """
        submodel1 = self.template.makeSubmodelTemplateName(1)
        model = """
            A: %s();
            species SI, SO
            SAI is A.%s
            SAO is A.%s
            SI -> XI; k1*SI
            XI -> SAI; k2*XI
            SAO -> SO; k3*SAO
            SO -> ; k4*SO
            SO -> XI; k5*SO
            k1 = %f
            k2 = %f
            k3 = %f
            k4 = %f
            k5 = %f
            """ % (submodel1,  self.input_name, self.output_name,
                   k1, k2, k3, k4, k5)
        s = control.TransferFunction.s
        numr = k1*k2*k3*(s + self.kO)*self.named_transfer_function.transfer_function
        denom = (s + self.kO + k3)*(s + self.kI)*(s + k4 + k5)*(s + k2)
        denom += k2*k3*k5*(s + self.kO)*self.named_transfer_function.transfer_function
        transfer_function = numr/denom
        # FIXME: -- handle operating region
        operating_region = self.operating_region
        kI = k1
        kO = k4 + k5
        network = SLMNetwork(model, "SI", "SO", kI, kO, transfer_function,
                              operating_region=operating_region, times=self.times,
                              children=[self])
        return network
    
    def amplify(self, k1, k2)->"SLMNetwork":
        """
        Creates a new network by amplifying the output of the current network. Let N.SI be the input species to the
        current network and N.SO be the output species from the current network. The new network will have the following reactions
        SI -> N.S; k1*SI
        N.S -> SO; k2*N.S
        """
        raise NotImplementedError("Must implement")