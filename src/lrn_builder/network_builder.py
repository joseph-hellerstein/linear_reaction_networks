'''Builds a sequential network with a degradation reaction at each stage.'''

"""
Conventions for Antimony models.
1. All reactions are either bounderies or uni-uni reactions.
2. Species are named S{num}, where num is the species index starting at 1. S1 is a boundary species.
3. Rate constants are named k{num}, where num is reactant_index + 100 * product_index. For
    a degradation reaction, it is just the product index.

Example:
    -> S1; k100
    S1 -> S2; k201*S1
    S2 -> ; k2*S2
    S2 -> S3; k302*S2
"""
from src.lrn_builder.make_symbolic_jacobian import makeSymbolicJacobian

from collections import namedtuple
import numpy as np
import pandas as pd # type: ignore
from scipy import signal  # type: ignore
import sympy as sp  # type: ignore
import tellurium as te # type: ignore
from typing import Tuple, Union, Optional

# Constants
REACTANT_FACTOR = 100  # Factor used to name rate constant (row in A matrix)
PRODUCT_FACTOR = 1   # Factor used to name rate constant (column in A matrix)


MakeSymbolicAMatResult = namedtuple("MakeSymbolicAMatResult", ["A_sym", "rate_dct"])


class NetworkBuilder(object):
    def __init__(self, num_species: int):
        self.num_species = num_species
        if self.num_species < 1:
            raise ValueError("num_species must be at least 1")
        self.antimon_str = None
        self.A_sym = None
        self.A_mat = None
        self.rate_dct = None # key is name, value is numeric rate constant
        self.transfer_function = None
        self.expression = None

    @staticmethod
    def _makeConstantName(product_index: int, reactant_index: int,
            constant_prefix: str = "k") -> str:
        # Product index is the column; reactant index is the row
        num = (1+reactant_index)*REACTANT_FACTOR + (1 + product_index)*PRODUCT_FACTOR
        return f"{constant_prefix}{num}"
    
    def makeRandomNetwork(self,
                num_reaction: int,
                kinetics_values: Tuple[float, float],
                num_product: Tuple[int, int],
                stoichiometry: Tuple[int, int]) -> str:
        """Creates a network with linear ODE kinetics: (a) at most 1 reactant (with unit stoichiometry);
        (b) mass action kinetics

        Returns:
            Antimony string
        """
        for idx in range(num_reaction):
            pass

#    @classmethod
#    def makeSymbolicAMat(cls, model: Union[str, np.ndarray], constant_prefix: str = "k") -> MakeSymbolicAMatResult:
#        """
#        Converts a numeric A matrix into a symbolic one using sympy symbols for the
#        rate constants. The underlying system is assumed to have a single reactant
#        and uses mass action kinetics.
#
#        A non-zero value in row i, column j of the A matrix indicates a reaction from species
#        Sj to Si. The value itself is not used, only its presence (non-zero) indicates that they connect. So, S2 -> S3 would have a rate constant named k203, 100 times the source
#        number plus the destination number.
#
#        The name assigned to each rate constant is based on the reactant and product indices.
#        if the reaction is S_j -> S_i, then the rate constant is named:
#            k{(1+j)*REACTANT_FACTOR + (1+i)*PRODUCT_FACTOR}
#
#        Args:
#            model: Numeric A matrix (numpy ndarray) or Antimony model string
#            constant_prefix: Prefix for the rate constant names
#            
#        Returns:
#            MakeSymbolicAMatResult: Contains the symbolic A matrix and the rate constant dictionary
#        """
#        if isinstance(model, np.ndarray):
#            A_mat = model
#        elif isinstance(model, str):
#            rr = te.loada(model)
#            A_mat = rr.getFullJacobian()
#        else:
#            raise TypeError("model must be a numpy ndarray or Antimony model string")
#
#        num_row, num_col = A_mat.shape
#        if num_row != num_col:
#            raise ValueError("A matrix must be square")
#        A_sym = sp.Matrix.zeros(num_row, num_col)
#        diagonal_sym = sp.Matrix.zeros(num_row)  # Sum of each reaction in which i is a reactant
#        # Convert off-diagonal elements to constants
#        rate_dct: dict = {}
#        for irow in range(num_row):
#            for icol in range(num_col):
#                value = A_mat[irow, icol]
#                if value == 0:
#                    continue
#                if irow == icol:
#                    continue
#                # Non-zero value means 
#                #   Reaction: S_icol -> S_irow
#                rate_constant_name = cls._makeConstantName(icol, irow,
#                        constant_prefix=constant_prefix)
#                rate_constant_sym = sp.symbols(rate_constant_name)
#                diagonal_sym[icol] += rate_constant_sym
#                A_sym[irow, icol] = rate_constant_sym  # Reaction increases S_irow
#                rate_dct[rate_constant_name] = value
#        # Set diagonal elements
#        for i in range(num_row):
#            rate_constant_name = f"{constant_prefix}{1+i}"
#            rate_constant_sym = sp.symbols(rate_constant_name)
#            A_sym[i, i] = -diagonal_sym[i] - rate_constant_sym  # Degradation reaction
#            rate_dct[rate_constant_name] = A_mat[i, i]
#        #
#        return MakeSymbolicAMatResult(A_sym=A_sym, rate_dct=rate_dct)

#    def makeRandomNetwork(self,
#                prob_reaction: float,
#                kinetic_range: Tuple[float, float],
#                constant_prefix: str = "k") -> None:
#        """
#        Generates a random Antimony model with a specified number of species and reactions.
#        S1 is the input and so is a boundary species.
#
#        Args:
#            num_species: Number of species in the model
#            prob_reaction: Probability of a reaction between any two species
#            kinetic_range: Tuple specifying the min and max values for kinetic constants
#
#        Updates
#            self.antimon_str: Random Antimony model string
#            self.A_sym: Symbolic A matrix
#            self.A_mat: Numeric A matrix
#            self.rate_dct: Dictionary of rate constants
#            self.transfer_function:
#        """
#        # Create a random A matrix
#        A_mat = np.random.uniform(kinetic_range[0], kinetic_range[1],
#                size=(self.num_species, self.num_species))
#        # Select a subset that are non-zero
#        for irow, icol in np.ndindex(A_mat.shape):
#            if irow == icol:
#                continue
#            if np.random.rand() > prob_reaction:
#                A_mat[irow, icol] = 0
#        # Ensure that species 1 is a boundary species (no reactions lead to it)
#        A_mat[0, :] = 0
#        # Make sure that degradations match the synthesis
#        diagonal_idx = np.arange(self.num_species)
#        A_mat[diagonal_idx, diagonal_idx] = 0
#        for i in range(self.num_species):
#            A_mat[i, i] = -np.sum(A_mat[:, i])
#        # Make the result symbolic with the correct rate constant names
#        A_sym = makeSymbolicJacobian(A_mat)
#        # Make the Antimony model
#        lines = []
#        lines.append("model *random_network()")
#        # Add reactions
#        rate_constant_names = []
#        specie_names = []
#        for irow, icol in np.ndindex(A_mat.shape):
#            value = A_mat[irow, icol]
#            if value == 0:
#                continue
#            if irow == icol:
#                continue
#            # Reaction from S_icol to S_irow
#            rate_constant_name = self._makeConstantName(icol, irow,
#                    constant_prefix=constant_prefix)
#            rate_constant_names.append(rate_constant_name)
#            reactant_name = f"S{icol}"
#            product_name = f"S{irow}"
#            specie_names.append(reactant_name)
#            specie_names.append(product_name)
#            lines.append(f"    {reactant_name} -> {product_name}; {rate_constant_name}*{reactant_name}")
#        # Initialize species
#        for species in set(specie_names):
#            if species == "S1":
#                lines.append(f"    $S1 = 1")
#            else:
#                lines.append(f"    {species} = 0")
#        # Define rate constants
#        for rate_constant in set(rate_constant_names):
#            lines.append(f"    {rate_constant} = {np.random.uniform(*kinetic_range)}")
#        # Complete the model
#        lines.append("end")
#        antimony_str = "\n".join(lines)
#        # Construct the transfer function
#        A_sym = self.makeSymbolicAMat(A_mat, constant_prefix=constant_prefix).A_sym
#        expression = A_sym[0, self.num_species]
#        transfer_function = self.sympyToTransferFunction(expression)
#        # Return Antimony string and symbolic A matrix
#        self.antimon_str = antimony_str
#        self.A_sym = A_sym
#        self.A_mat = A_mat
#        self.rate_dct = self.makeSymbolicAMat(A_mat,
#        return MakeRandomNetworkResult(antimony_str=antimony_str,
#                input_name="S1", output_name=f"S{self.num_species}", transfer_function=transfer_function)

    def makeSequentialAntimony(self) -> str:
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
        for i in range(self.num_species):
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
        for i in range(self.num_species + 1):
            lines.append(f"    S{i} = 0")
        
        lines.append("end")
        
        return "\n".join(lines)

    def makeTransferFunction(self, output_name : Optional[str] = None, **kwargs) -> signal.TransferFunction:
        """
        Uses the current symbolic A matrix to construct a transfer function from S1 to
        the largest numbered species on output.
        
        Parameters:
        -----------
        output_name: Species number to output the transfer function for (default: largest numbered species)
        kwargs: name-value pairs for parameters

        Returns:
        --------
        signal.TransferFunction
            Transfer function object
        """
        if self.A_sym is None:
            raise ValueError("Symbolic A matrix is not defined")
        if output_name is None:
            output_name = self.A_mat.colnames[-1]
        output_index = list(A_mat.colnames).index(output_name)
        # Construct the transfer function
        s, u = sp.symbols(["s", "u"])
        # FIXME to size B
        B = sp.Matrix([[1], [0], [0], [0], [0]])
        I = sp.eye(self.A_sym.rows)
        species_transform_vec = (s*I - self.A_sym).inv() * B
        #
        expr = self.species_transform_vec[output_index] # Output signal
        change_dct = dict(self.rate_constant)
        change_dct.update(kwargs)
        expr = expr.subs(change_dct)
        # All free symbols except for s should be removed. Now convert to transfer function
        free_syms = expr.free_symbols
        if len(free_syms) == 0:
            raise ValueError("Expression contains no symbols (it's just a constant)")
        elif len(free_syms) > 1:
            raise ValueError(f"Expression contains multiple symbols: {free_syms}. "
                            "Only single-variable rational functions are supported.")
        s = free_syms.pop()
        # Simplify and get numerator and denominator
        expr = sp.simplify(expr)
        numer, denom = sp.fraction(expr)
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