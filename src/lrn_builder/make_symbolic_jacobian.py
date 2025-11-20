'''Construct a symbolic Jacobian matrix from an Antimony model.'''

from collections import namedtuple
import libsbml # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from collections import OrderedDict

# FIXME: Should the calculation of degradation reactions be considered in calculating symbolic Jacobian?


MakeSymbolicJacobianResult = namedtuple("MakeSymbolicJacobianResult",
        ["jacobian_mat", "jacobian_smat", "kinetic_constant_dct"])
#  jacobian_mat : sympy.Matrix
#      Symbolic Jacobian matrix where J[i,j] = d(rate_i)/d(spec
#  jacobian_smat : sympy.Matrix
#      Sparse symbolic Jacobian matrix
#  kinetic_constant_dict : dict
#      Dictionary mapping parameter names to their values

def makeSymbolicJacobian(antimony_str: str) -> MakeSymbolicJacobianResult:
    """
    Convert an Antimony model to a symbolic Jacobian matrix.
    Reactions must either have 0 or 1 reactant with unit stoichiometry.
    Kinetics are assumed to be mass action (k*S) if 1 reactant
    
    Parameters:
    -----------
    antimony_str: String for an Antimony model
        Path to Antimony file
        
    Returns:
    --------
    jacobian : sympy.Matrix
        Symbolic Jacobian matrix where J[i,j] = d(rate_i)/d(species_j)
    param_dict : dict
        Dictionary mapping parameter names to their values
        
    Notes:
    ------
    Assumes reactions have 0 or 1 reactant with unit stoichiometry.
    Kinetics are mass action (k*S) if 1 reactant, or fixed rate (k) if 0 reactants.
    """
    # Load with tellurium and convert to SBML
    rr = te.loada(antimony_str)
    jacobian_mat = rr.getFullJacobian()
    sbml_str = rr.getSBML()
    # Get the SBML model
    reader = libsbml.SBMLReader()
    document = reader.readSBMLFromString(sbml_str)
    if document.getNumErrors() > 0:
        print("Errors in SBML conversion:")
        document.printErrors()
        raise ValueError("Failed to convert Antimony to SBML")
    model = document.getModel()
    # Get species and create symbols
    species_list = []
    species_symbols = {}
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        species_id = species.getId()
        # Skip boundary/constant species
        #if not species.getBoundaryCondition() and not species.getConstant():
        species_list.append(species_id)
        species_symbols[species_id] = sp.Symbol(species_id)
    # Get parameters and create symbols and dictionary
    kinetic_constant_dct = {}
    param_symbols = {}
    for i in range(model.getNumParameters()):
        param = model.getParameter(i)
        param_id = param.getId()
        param_value = param.getValue()
        kinetic_constant_dct[param_id] = param_value
        param_symbols[param_id] = sp.Symbol(param_id)
    # Get local parameters from reactions. Construct symbols.
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        kinetic_law = reaction.getKineticLaw()
        if kinetic_law:
            for j in range(kinetic_law.getNumParameters()):
                param = kinetic_law.getParameter(j)
                param_id = param.getId()
                param_value = param.getValue()
                kinetic_constant_dct[param_id] = param_value
                param_symbols[param_id] = sp.Symbol(param_id)
    # Construct rate equations for each species
    rate_equation_dct = {species_id: sp.Integer(0) for species_id in species_list}
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        reaction_id = reaction.getId()
        # Get reactants (should be 0 or 1)
        num_reactants = reaction.getNumReactants()
        if num_reactants > 1:
            raise ValueError(f"Reaction {reaction_id} has {num_reactants} reactants. Only 0 or 1 allowed.")
        reactant_id = None
        if num_reactants == 1:
            reactant = reaction.getReactant(0)
            if reactant.getStoichiometry() != 1:
                raise ValueError(f"Reaction {reaction_id} has non-unit stoichiometry")
            reactant_id = reactant.getSpecies()
        # Get products
        product_stoichiometry = {}
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            product_id = product.getSpecies()
            stoich = product.getStoichiometry()
            product_stoichiometry[product_id] = stoich
        # Get rate constant from kinetic law
        kinetic_law = reaction.getKineticLaw()
        if not kinetic_law:
            raise ValueError(f"Reaction {reaction_id} has no kinetic law")
        # Parse the kinetic law to extract the rate constant
        math_ast = kinetic_law.getMath()
        rate_formula_str = libsbml.formulaToL3String(math_ast)
        # Extract rate constant and build mass action rate
        rate_constant_symbol = extract_rate_constant(
            rate_formula_str, 
            reactant_id, 
            param_symbols, 
            species_symbols,
            reaction_id
        )
        # Build rate expression based on mass action kinetics
        if reactant_id is None:
            # No reactant: fixed rate k
            rate_expr = rate_constant_symbol
        else:
            # One reactant: mass action k*S
            rate_expr = rate_constant_symbol * species_symbols[reactant_id]
        # Update rate equations
        # Reactant is consumed
        if reactant_id and reactant_id in species_list:
            rate_equation_dct[reactant_id] -= rate_expr
        # Products are produced
        for product_id, stoich in product_stoichiometry.items():
            if product_id in species_list:
                rate_equation_dct[product_id] += stoich * rate_expr
    # Create ODE matrix (column vector)
    ode_matrix = sp.Matrix([rate_equation_dct[species_id] for species_id in species_list])
    # Compute Jacobian matrix
    # J[i,j] = d(f_i)/d(x_j) where f_i is the rate equation for species i
    species_vector = sp.Matrix([species_symbols[species_id] for species_id in species_list])
    jacobian_smat = ode_matrix.jacobian(species_vector)
    #
    return MakeSymbolicJacobianResult(
        jacobian_mat=jacobian_mat,
        jacobian_smat = jacobian_smat,
        kinetic_constant_dct = kinetic_constant_dct)


def extract_rate_constant(formula_str, reactant_id, param_symbols, species_symbols, reaction_id):
    """
    Extract the rate constant from the kinetic law formula.
    
    Expects either:
    - "k" for zero-reactant reactions
    - "k * S" or "S * k" for single-reactant reactions
    
    Parameters:
    -----------
    formula_str : str
        SBML formula as string
    reactant_id : str or None
        Species ID of the reactant (None if no reactant)
    param_symbols : dict
        Mapping from parameter IDs to SymPy symbols
    species_symbols : dict
        Mapping from species IDs to SymPy symbols
    reaction_id : str
        Reaction ID for error messages
        
    Returns:
    --------
    rate_constant : sympy.Symbol
        The rate constant symbol
    """
    # Parse the formula
    all_symbols = {**species_symbols, **param_symbols}
    formula_str = formula_str.replace('^', '**')
    
    try:
        expr = sp.sympify(formula_str, locals=all_symbols)
    except Exception as e:
        raise ValueError(f"Error parsing kinetic law for reaction {reaction_id}: {formula_str}") from e
    
    if reactant_id is None:
        # Zero reactant: formula should just be the rate constant
        if expr in param_symbols.values():
            return expr
        else:
            # The expression might be a number or more complex
            # Try to find a parameter in it
            params_in_expr = [sym for sym in expr.free_symbols if sym in param_symbols.values()]
            if len(params_in_expr) == 1:
                return params_in_expr[0]
            elif len(params_in_expr) == 0:
                raise ValueError(f"Reaction {reaction_id}: No parameter found in kinetic law {formula_str}")
            else:
                raise ValueError(f"Reaction {reaction_id}: Multiple parameters in kinetic law {formula_str}")
    else:
        # One reactant: formula should be k*S
        # Divide by the reactant to get the rate constant
        reactant_symbol = species_symbols[reactant_id]
        
        # Try to extract rate constant by dividing
        if reactant_symbol in expr.free_symbols:
            # Simplify k*S / S = k
            rate_constant = sp.simplify(expr / reactant_symbol)
            
            # Check that the result doesn't contain the reactant anymore
            if reactant_symbol in rate_constant.free_symbols:
                raise ValueError(f"Reaction {reaction_id}: Kinetic law {formula_str} is not mass action")
            
            return rate_constant
        else:
            raise ValueError(f"Reaction {reaction_id}: Kinetic law {formula_str} doesn't contain reactant {reactant_id}")