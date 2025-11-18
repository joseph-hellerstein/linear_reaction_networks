'''Builds a sequential network with a degradation reaction at each stage.'''

import numpy as np
import sympy as sp  # type: ignore
import tellurium as te # type: ignore
import pandas as pd # type: ignore
from typing import Tuple, Optional

# Constants
REACTANT_FACTOR = 100  # Factor used to name rate constant
PRODUCT_FACTOR = 1   # Factor used to name rate constant



class NetworkBuilder(object):
    def __init__(self, num_species: int):
        self.num_species = num_species
        if self.num_species < 1:
            raise ValueError("num_species must be at least 1")

    @staticmethod
    def _makeConstantName(reactant_index: int, product_index: int,
            constant_prefix: str = "k") -> str:
        num = (1+reactant_index)*REACTANT_FACTOR + (1 + product_index)*PRODUCT_FACTOR
        return f"{constant_prefix}{num}"

    @classmethod
    def makeSymbolicAMatrix(cls, A_mat: np.ndarray, constant_prefix: str = "k") -> sp.Matrix:
        """
        Converts a numeric A matrix into a symbolic one using sympy symbols for the
        rate constants. The underlying system is assumed to have a single reactant
        and uses mass action kinetics.

        A non-zero value in row i, column j of the A matrix indicates a reaction from species
        Sj to Si. The value itself is not used, only its presence (non-zero) indicates that they connect. So, S2 -> S3 would have a rate constant named k203, 100 times the source
        number plus the destination number.

        The name assigned to each rate constant is based on the reactant and product indices.
        if the reaction is S_j -> S_i, then the rate constant is named:
            k{(1+j)*REACTANT_FACTOR + (1+i)*PRODUCT_FACTOR}

        Args:
            numeric_A_matrix: Numeric A matrix (numpy ndarray)
            constant_prefix: Prefix for the rate constant names
            
        Returns:
            sympy.Matrix: Symbolic A matrix
        """
        num_row, num_col = A_mat.shape
        if num_row != num_col:
            raise ValueError("A matrix must be square")
        A_sym = sp.Matrix.zeros(num_row, num_col)
        diagonal_sym = sp.Matrix.zeros(num_row)  # Sum of each reaction in which i is a reactant
        # Convert off-diagonal elements to constants
        for irow in range(num_row):
            for icol in range(num_col):
                value = A_mat[irow, icol]
                if value == 0:
                    continue
                if irow == icol:
                    continue
                # Non-zero value means 
                #   Reaction: S_icol -> S_irow
                rate_constant_name = cls._makeConstantName(icol, irow,
                        constant_prefix=constant_prefix)
                rate_constant_sym = sp.symbols(rate_constant_name)
                diagonal_sym[icol] += rate_constant_sym
                A_sym[irow, icol] = rate_constant_sym  # Reaction increases S_irow
        # Set diagonal elements
        for i in range(num_row):
            A_sym[i, i] = -diagonal_sym[i]
        #
        return A_sym

    def makeRandomNetwork(self,
                prob_reaction: float,
                kinetic_range: Tuple[float, float],
                constant_prefix: str = "k") -> Tuple[str, sp.Matrix]:
        """
        Generates a random Antimony model with a specified number of species and reactions.
        S1 is the input and so is a boundary species.

        Args:
            num_species: Number of species in the model
            prob_reaction: Probability of a reaction between any two species
            kinetic_range: Tuple specifying the min and max values for kinetic constants

        Returns:
            str: Random Antimony model string
        """
        # Create a random A matrix
        A_mat = np.random.uniform(kinetic_range[0], kinetic_range[1],
                size=(self.num_species, self.num_species))
        # Select a subset that are non-zero
        for irow, icol in np.ndindex(A_mat.shape):
            if irow == icol:
                continue
            if np.random.rand() > prob_reaction:
                A_mat[irow, icol] = 0
        # Ensure that species 1 is a boundary species (no reactions lead to it)
        A_mat[0, :] = 0
        # Make sure that degradations match the synthesis
        diagonal_idx = np.arange(self.num_species)
        A_mat[diagonal_idx, diagonal_idx] = 0
        for i in range(self.num_species):
            A_mat[i, i] = -np.sum(A_mat[:, i])
        # Make the result symbolic with the correct rate constant names
        A_sym = self.makeSymbolicAMatrix(A_mat)
        # Make the Antimony model
        lines = []
        lines.append("model *random_network()")
        # Add reactions
        rate_constant_names = []
        specie_names = []
        for irow, icol in np.ndindex(A_mat.shape):
            value = A_mat[irow, icol]
            if value == 0:
                continue
            if irow == icol:
                continue
            # Reaction from S_icol to S_irow
            rate_constant_name = self._makeConstantName(icol, irow,
                    constant_prefix=constant_prefix)
            rate_constant_names.append(rate_constant_name)
            reactant_name = f"S{icol}"
            product_name = f"S{irow}"
            specie_names.append(reactant_name)
            specie_names.append(product_name)
            lines.append(f"    {reactant_name} -> {product_name}; {rate_constant_name}*{reactant_name}")
        # Initialize species
        for species in set(specie_names):
            lines.append(f"    {species} = 0")
        # Define rate constants
        for rate_constant in set(rate_constant_names):
            lines.append(f"    {rate_constant} = {np.random.uniform(*kinetic_range)}")
        # Complete the model
        lines.append("end")
        antimony_str = "\n".join(lines)
        # Return Antimony string and symbolic A matrix
        return antimony_str, A_sym

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