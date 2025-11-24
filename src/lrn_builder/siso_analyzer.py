'''Does SISO analysis of an LTI Antimony model.'''

"""
Conventions for Antimony models.
1. All reactions are either bounderies or uni-uni reactions.
2. Species are named S{num}, where num is the species index starting at 1. S1 is a boundary species.
"""


from src.lrn_builder.make_symbolic_jacobian import makeSymbolicJacobian

from collections import namedtuple
import numpy as np
import pandas as pd # type: ignore
from scipy import signal  # type: ignore
import sympy as sp  # type: ignore
import tellurium as te # type: ignore
from typing import Tuple, Union, Optional, cast, Any

# Constants
REACTANT_FACTOR = 100  # Factor used to name rate constant (row in A matrix)
PRODUCT_FACTOR = 1   # Factor used to name rate constant (column in A matrix)
INPUT_NAME = "S1"  # Default input species nJame
# Null values for key attributes
NULL_JACOBIAN_SMAT = sp.Matrix()
NULL_JACOBIAN_DF = pd.DataFrame()
NULL_ANTIMONY_STR = ""
NULL_KINETIC_CONSTANT_DCT = {}
NULL_TRANSFER_FUNCTION_EXPR = sp.Symbol("Unknown")
NULL_TRANSFER_FUNCTION_MATRIX = sp.Matrix()
NULL_SPECIES_NAME = ""
DEFAULT_INPUT_NAME = "S1_"


MakeSymbolicAMatResult = namedtuple("MakeSymbolicAMatResult", ["A_sym", "rate_dct"])


class SISOAnalyzer(object):
    def __init__(self, antimony_str: str,
            input_name:str = DEFAULT_INPUT_NAME,
            output_name:str = NULL_SPECIES_NAME) -> None:
        """
        Initializes the NetworkAnalyzer with an optional Antimony model string.

        Parameters:
        -----------
        antimony_str: Antimony model string
            Antimony model string to initialize the NetworkAnalyzer
        input_name: Species name to use as input (default: S1)
        output_name: Species name to use as output (default: largest numbered species)

        Returns:
        --------
        None
        """
        # Initialize to the correct type. Changing self.antimony_str triggers
        # recalculation of jacobian matrices and kinetic_constant_dct
        self.original_antimony_str = antimony_str
        self.input_name = input_name
        self.output_name = output_name
        self.antimony_str = antimony_str.replace(f"${input_name}", f"{input_name}")  # Remove boundary marker for Tellurium
        self.roadrunner = te.loada(self.antimony_str)
        self._jacobian_df = NULL_JACOBIAN_DF
        self._species_names = []
        self._jacobian_smat = NULL_JACOBIAN_SMAT
        self._kinetic_constant_dct = NULL_KINETIC_CONSTANT_DCT
        self._transfer_function_smat = NULL_TRANSFER_FUNCTION_MATRIX
        self._transfer_function_expr = NULL_TRANSFER_FUNCTION_EXPR
        self._updateOutputName()

    def _updateOutputName(self) -> None:
        """
        Updatesthe input and output species names.

        Returns:
        --------
        Tuple[str, str]
            (input_name, output_name)
        """
        candidate_names = self.roadrunner.getExtendedStoichiometryMatrix().rownames
        if not self.input_name in candidate_names:
            raise ValueError(f"Input species {self.input_name} not found in model species: {candidate_names}")
        if self.output_name == NULL_SPECIES_NAME:
            candidate_names = cast(list[str], self.jacobian_df.columns)  # type: ignore
            self.output_name = candidate_names[-1]
        elif not self.output_name in candidate_names:
            raise ValueError(f"Output species {self.output_name} not found in model species: {candidate_names}")

########### GETTERS AND SETTERS #############
    @property
    def jacobian_df(self) -> pd.DataFrame:
        """Returns the numeric Jacobian DataFrame."""
        if len(self._jacobian_df) == 0:
            jacobian_mat = self.roadrunner.getFullJacobian()
            column_names = cast(list[str], jacobian_mat.colnames)  # type: ignore
            sorted_column_idxs = np.argsort(column_names)
            sorted_species_names = [column_names[i] for i in sorted_column_idxs]
            arrs = []
            for idx, name in enumerate(sorted_species_names):
                new_idx = sorted_column_idxs[idx]
                arr = jacobian_mat[new_idx, sorted_column_idxs]
                arrs.append(arr)
            sorted_arr = np.array(arrs)
            self._jacobian_df = pd.DataFrame(sorted_arr,
                                index=sorted_species_names,
                                columns=sorted_species_names)
        return self._jacobian_df

    """ 
    @property
    def jacobian_mat(self) -> np.ndarray:
        # Returns the symbolic Jacobian matrix.
        if len(self._jacobian_mat) > 0:
            return self._jacobian_mat
        self._jacobian_mat = self.roadrunner.getFullJacobian()
        return self._jacobian_mat
    """
    
    @property
    def species_names(self) -> list[str]:
        """Returns the list of species names in the model."""
        if len(self._species_names) == 0:
            self._species_names = list(self.jacobian_df.columns)  # type: ignore
        return self._species_names
    
    @property
    def num_species(self) -> int:
        """Returns the number of species in the model."""
        return len(self.species_names)
    
    @property
    def jacobian_smat(self) -> sp.Matrix:
        """Returns the symbolic Jacobian matrix."""
        if len(self._jacobian_smat) > 0:
            return self._jacobian_smat
        self._updateJacobianAndKineticConstants()
        return self._jacobian_smat
    
    @property
    def kinetic_constant_dct(self) -> dict:
        """Returns the symbolic Jacobian matrix."""
        if len(self._kinetic_constant_dct) > 0:
            return self._kinetic_constant_dct
        self._updateJacobianAndKineticConstants()
        return self._kinetic_constant_dct
    
    @property
    def transfer_function_smat(self) -> sp.Matrix:
        """Returns the symbolic transfer function expression."""
        if self._transfer_function_expr == NULL_TRANSFER_FUNCTION_EXPR:
            # Construct the transfer function expression
            input_index = self.species_names.index(self.input_name)
            s = sp.Symbol("s")
            B = sp.Matrix(np.zeros((self.num_species, self.num_species)))
            B[input_index, input_index] = 1
            I = sp.eye(self.jacobian_smat.rows)
            self._transfer_function_smat = (s*I - self.jacobian_smat).inv() * B
        return self._transfer_function_smat
    
    @property
    def transfer_function_expr(self)->Any:
        """Returns the symbolic transfer function expression."""
        if self._transfer_function_expr == NULL_TRANSFER_FUNCTION_EXPR:
            # Construct the transfer function expression
            input_index = self.species_names.index(self.input_name)
            output_index = self.species_names.index(self.output_name)
            self._transfer_function_expr = self.transfer_function_smat[output_index, input_index]
        return self._transfer_function_expr
    
    def _updateJacobianAndKineticConstants(self) -> None:
        """Updates the Jacobian matrices and kinetic constant dictionary."""
        result = makeSymbolicJacobian(self.antimony_str)
        self._jacobian_smat = result.jacobian_smat
        self._kinetic_constant_dct = result.kinetic_constant_dct

    ############ METHODS #############

    def makeSequentialAntimony(self, num_species) -> str:
        """
        Generates an Antimony file consisting of num_stage sequences of uni-uni reactions 
        and a degradation reaction. The kinetic constants have sequential values and are 
        labelled "k1", "k2", "k3", etc. with values 1, 2, 3, etc.
        
        Args:
            num_stage: Number of stages in the sequential network
            
        Returns:
            str: Antimony model string
            
        Example:
            For num_stage=3:
            - S0 -> S1; k1*S0   (k1 = 1)
            - S1 -> ; k2*S1     (k2 = 2)
            - S1 -> S2; k3*S1   (k3 = 3)
            - S2 -> ; k4*S2     (k4 = 4)
            - S2 -> S3; k5*S2   (k5 = 5)
            - S3 -> ; k6*S3     (k6 = 6)
        """

        lines = []
        lines.append("model *sequential_network()")
        
        # Generate reactions and parameters for each stage
        k_counter = 1
        for i in range(num_species):
            source_species = f"S{i}"
            target_species = f"S{i+1}"
            
            # Uni-uni reaction: Si -> Si+1
            forward_rate_name = f"k{k_counter}"
            lines.append(f"    {source_species} -> {target_species}; {forward_rate_name}*{source_species}")
            k_counter += 1
            
            # Degradation reaction: Si -> ;
            deg_rate_name = f"k{k_counter}"
            lines.append(f"    {source_species} -> ; {deg_rate_name}*{source_species}")
            k_counter += 1
        
        lines.append("")
        
        # Define kinetic constants
        for i in range(1, k_counter):
            lines.append(f"    k{i} = {i}")
        
        lines.append("")
        
        # Initialize species concentrations
        for i in range(num_species + 1):
            lines.append(f"    S{i} = 0")
        
        lines.append("end")
        
        return "\n".join(lines)

    def makeTransferFunction(self, **kwargs) -> signal.TransferFunction:
        """
        Uses the current symbolic A matrix to construct a transfer function from S1 to
        the largest numbered species on output.
        
        Parameters:
        -----------
        kwargs: name-value pairs for parameters

        State updates
        ----------
        self.kinetic_constant_dct: Dictionary of kinetic constants
        self.transfer_function_expr: Symbolic transfer function expression
        self.transfer_function: sign.TransferFunction object

        Returns:
        --------
        signal.TransferFunction
            Transfer function object
        """
        kinetic_constant_dct = dict(self.kinetic_constant_dct)
        kinetic_constant_dct.update(kwargs)
        transfer_function_expr = self.transfer_function_expr.subs(kinetic_constant_dct)
        # All free symbols except for s should be removed. Now convert to transfer function
        free_syms = transfer_function_expr.free_symbols
        if len(free_syms) == 0:
            return signal.TransferFunction([float(transfer_function_expr)], [1])
        elif len(free_syms) > 1:
            raise ValueError(f"Expression contains multiple symbols: {free_syms}. "
                            "Only single-variable rational functions are supported.")
        s = free_syms.pop()
        # Simplify and get numerator and denominator
        transfer_function_expr = sp.simplify(transfer_function_expr)
        numer, denom = sp.fraction(transfer_function_expr)
        # Convert to polynomials
        numer_poly = sp.Poly(numer, s)
        denom_poly = sp.Poly(denom, s)
        # Extract coefficients (from highest to lowest degree)
        numer_coeffs = [float(c) for c in numer_poly.all_coeffs()]
        denom_coeffs = [float(c) for c in denom_poly.all_coeffs()]
        # Create transfer function
        self.transfer_function = signal.TransferFunction(numer_coeffs, denom_coeffs) 
        #
        return self.transfer_function