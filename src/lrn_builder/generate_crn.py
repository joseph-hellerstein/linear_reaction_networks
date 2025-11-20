'''Generates random chemical reaction networks in Antimony language.'''

"""
Characteristics of the generated CRN.
1. Behaves as a linear system of ODEs.
2. Each reaction has one reactant and one or more products.
3. Kinetic constants are randomly assigned within specified bounds.
4. Stoichiometry of products is randomly assigned within specified bounds.
5. The first reaction has no reactant (i.e., it is a source reaction).
6. Subsequent reactions select reactants from existing species to ensure connectivity.
7. Species are named sequentially (S1, S2, S3, ...).
8. Kinetic constants are named sequentially (k1, k2, k3,
9. Species S1 is an input boundary.
"""

import random
import numpy as np
import tellurium as te  # type: ignore

def generateCrn(
    num_reactions,
    num_products_bounds=(1, 3),
    kinetic_constant_bounds=(0.1, 10.0),
    stoichiometry_bounds=(1, 3),
    seed=None
):
    """
    Generate a random chemical reaction network in Antimony language.
    
    Parameters:
    -----------
    num_reactions : int
        Total number of reactions to generate
    num_products_bounds : tuple of (int, int)
        (min, max) number of products per reaction
    kinetic_constant_bounds : tuple of (float, float)
        (min, max) values for kinetic rate constants
    stoichiometry_bounds : tuple of (int, int)
        (min, max) stoichiometry coefficients for product species
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    str : Antimony model string
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if num_reactions < 1:
        raise ValueError("Must have at least 1 reaction")
    
    antimony_lines = []
    antimony_lines.append("# Random Chemical Reaction Network")
    antimony_lines.append("# Generated with specified constraints\n")
    antimony_lines.append("model random_crn()\n")
    
    # Track existing species and rate constants
    existing_species = set()
    species_counter = 1
    rate_constants = []
    
    # Reaction 1: No reactant, S1 is product
    k1 = np.random.uniform(kinetic_constant_bounds[0], kinetic_constant_bounds[1])
    rate_constants.append(("k1", k1))
    antimony_lines.append(f"  # Reaction 1")
    antimony_lines.append(f"  -> S1; k1")
    existing_species.add("S1")
    species_counter = 2
    
    # Generate subsequent reactions
    for rxn_idx in range(2, num_reactions + 1):
        antimony_lines.append(f"\n  # Reaction {rxn_idx}")
        
        # Pick a random reactant from existing species
        reactant = random.choice(list(existing_species))
        
        # Determine number of products
        num_products = random.randint(num_products_bounds[0], num_products_bounds[1])
        
        # Generate products
        products = []
        for _ in range(num_products):
            # Decide whether to use existing species or create new one
            # Bias towards creating new species early on, existing species later
            if len(existing_species) < 3 or random.random() < 0.5:
                # Create new species
                new_species = f"S{species_counter}"
                existing_species.add(new_species)
                species_counter += 1
                product_species = new_species
            else:
                # Use existing species
                product_species = random.choice(list(existing_species))
            
            # Assign stoichiometry
            stoich = random.randint(stoichiometry_bounds[0], stoichiometry_bounds[1])
            products.append((stoich, product_species))
        
        # Generate kinetic constant
        k = np.random.uniform(kinetic_constant_bounds[0], kinetic_constant_bounds[1])
        k_name = f"k{rxn_idx}"
        rate_constants.append((k_name, k))
        
        # Build reaction string with spaces between stoichiometry and species
        product_str = " + ".join([f"{s} {sp}" if s > 1 else sp for s, sp in products])
        rate_law = f"{k_name} * {reactant}"
        
        antimony_lines.append(f"  {reactant} -> {product_str}; {rate_law}")
    
    # Add rate constant definitions
    antimony_lines.append("\n  # Rate constants")
    for k_name, k_value in rate_constants:
        antimony_lines.append(f"  {k_name} = {k_value:.4f}")
    
    # Add species initialization
    antimony_lines.append("\n  # Species initialization")
    for species in sorted(existing_species, key=lambda x: int(x[1:])):
        antimony_lines.append(f"  {species} = 0")
    
    antimony_lines.append("\nend")
    antimony_str = "\n".join(antimony_lines)

    # Make the network stable by adding degradation reactions.
    # We will use Gershgorin's Circle Theorem to check stability.
    rr = te.loada(antimony_str)
    jacobian = rr.getFullJacobian()
    eigenvalues = np.linalg.eigvals(jacobian)
    if np.any(eigenvalues.real >= 0):
        # Find the columns in the jacobian where the sum of the absolute values of the non-diagonal entries
        # in the same row is greater than or equal to the absolute value of the diagonal entry.
        degradation_dct = {}
        for i in range(jacobian.shape[0]):
            diag = abs(jacobian[i, i])
            off_diag_sum = np.sum(np.abs(jacobian[i, :])) - diag
            if off_diag_sum >= diag:
                margin = off_diag_sum - diag
                degradation_dct[i] = 1.1*margin
        # Add degradation reactions for each species
        degradation_lines = []
        for idx, degradation_rate in degradation_dct.items():
            species = f"S{idx + 1}"
            degradation_lines.append(f"  {species} -> ; kd_{species} * {species}")
            degradation_lines.append(f"  kd_{species} = {degradation_rate:.4f}")
        # Insert degradation reactions before 'end'
        antimony_lines = antimony_str.splitlines()
        antimony_lines.insert(-1, "\n  # Degradation reactions")
        antimony_lines[-1:-1] = degradation_lines
        antimony_str = "\n".join(antimony_lines)
    
    return antimony_str