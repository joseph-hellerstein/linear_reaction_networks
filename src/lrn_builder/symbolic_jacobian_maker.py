'''Creates a symbolic Jacobian matrix from an Antimony model.'''

from collections import namedtuple
import libsbml # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from collections import OrderedDict
from typing import Tuple, Dict, List


ReactionDescription = namedtuple("ReactionDescription",
        ["reaction_name", "reactant_name", "kinetic_constant_name", "product_stoichiometry_dct"]) 
MakeSymbolicJacobianResult = namedtuple("MakeSymbolicJacobianResult",
        ["jacobian_smat", "kinetic_constant_dct", "species_names"])
#  jacobian_mat : sympy.Matrix
#      Symbolic Jacobian matrix where J[i,j] = d(rate_i)/d(spec
#  jacobian_smat : sympy.Matrix
#      Sparse symbolic Jacobian matrix
#  kinetic_constant_dict : dict
#      Dictionary mapping parameter names to their values

class SymbolicJacobianMaker(object):

    def __init__(self, antimony_str: str):
        """
        Convert an Antimony model to a symbolic Jacobian matrix.
        Reactions must either have 0 or 1 reactant with unit stoichiometry.
        Kinetics are assumed to be mass action (k*S) if 1 reactant
        
        Parameters:
        -----------
        antimony_str: String for an Antimony model
            Path to Antimony file
        """
        self.antimony_str = antimony_str
        # See build() for other attributes

    def build(self)->None:
        # Workflow to build the symbolic Jacobian
        self.model = self._makeModel()
        self.kinetic_constant_dct: dict = self._makeKineticConstantDct()
        self.species_names: list = self._makeSpeciesNames()
        self.reaction_description_dct: dict = self._makeReactionDescription()
        self.jacobian_smat = self._makeSymbolicJacobian()

    def _getSpeciesIndex(self, species_name: str)->int:
        # Get the index of a species in the species_names list
        try:
            idx = self.species_names.index(species_name)
        except ValueError:
            raise ValueError(f"Species {species_name} not found in species names")
        return idx

    def _makeModel(self)->libsbml.Model:
        # Load with tellurium and convert to SBML
        rr = te.loada(self.antimony_str)
        sbml_str = rr.getSBML()
        # Get the SBML model
        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromString(sbml_str)
        if document.getNumErrors() > 0:
            print("Errors in SBML conversion:")
            document.printErrors()
            raise ValueError("Failed to convert Antimony to SBML")
        model = document.getModel()
        return model
    
    def _makeSpeciesNames(self)->List[str]:
        # Include all species, including boundary/constant species
        species_names = []
        # Get species and create symbols
        for i in range(self.model.getNumSpecies()):
            species = self.model.getSpecies(i)
            species_name = species.getId()
            species_names.append(species_name)
        species_names.sort()
        return species_names
    
    def _makeKineticConstantDct(self)->Dict[str, float]:
        kinetic_constant_dct = {}
        # Get parameters and create symbols and dictionary
        for i in range(self.model.getNumParameters()):
            param = self.model.getParameter(i)
            param_id = param.getId()
            param_value = param.getValue()
            kinetic_constant_dct[param_id] = param_value
        return kinetic_constant_dct

    def _makeReactionDescription(self)->Dict[str, ReactionDescription]:
        # Extract the rate constant and species for each reaction
        ##
        def findStrs(candidates: List[str], targets: List[str]) -> List[str]:
            # Find the candidates that are equal to the targets
            results = set(candidates).intersection(targets)
            return list(results)
        ##
        reaction_description_dct: Dict[str, ReactionDescription] = {}
        for i in range(self.model.getNumReactions()):
            reaction = self.model.getReaction(i)
            reaction_name = reaction.getId()
            # Get reactants (should be 0 or 1)
            num_reactants = reaction.getNumReactants()
            if num_reactants > 1:
                raise ValueError(f"Reaction {reaction_name} has {num_reactants} reactants. Only 0 or 1 allowed.")
            reactant_name = None
            if num_reactants == 1:
                reactant = reaction.getReactant(0)
                if reactant.getStoichiometry() != 1:
                    raise ValueError(f"Reaction {reaction_name} has non-unit stoichiometry")
                reactant_name = reactant.getSpecies()
            # Parse the kinetic law to extract the rate constant, products, and their stoichiometries
            kinetic_law_str = reaction.getKineticLaw()
            kinetic_law_strs = kinetic_law_str.formula.split(" ").strip()
            kinetic_constants = findStrs(kinetic_law_strs, list(self.kinetic_constant_dct.keys()))
            if len(kinetic_constants) != 1:
                raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
            kinetic_constant_name = kinetic_constants[0]
            species_names = findStrs(kinetic_law_strs, self.species_names)
            if len(species_names) > 1:
                raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
            # Get products and their stoichiometries
            product_stoichiometry_dct: dict = {}  # key: product_id, value: stoichiometry
            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                product_id = product.getSpecies()
                product_stoichiometry_dct[product_id] = product.getStoichiometry()
            # Create ReactionDescription
            reaction_description_dct[reaction_name] = ReactionDescription(
                reaction_name=reaction_name,
                reactant_name=reactant_name,
                kinetic_constant_name=kinetic_constant_name,
                product_stoichiometry_dct=product_stoichiometry_dct)
        #
        return reaction_description_dct
    
    def _makeSymbolicJacobian(self)->sp.Matrix:
        # Create the symbolic Jacobian matrix
        raise NotImplementedError("Not yet implemented")
    
    @classmethod
    def makeSymbolicJacobian(cls, antimony_str: str)->sp.Matrix:
        '''Create a symbolic Jacobian matrix from an Antimony model string.
        
        Parameters:
        -----------
        antimony_str: str
            Antimony model string
        
        Returns:
        --------
        MakeSymbolicJacobianResult
            Named tuple with fields:
            - jacobian_smat : sympy.Matrix
                Symbolic Jacobian matrix where J[i,j] = d(rate_i)/d(species_j)
            - kinetic_constant_dct : dict
                Dictionary mapping parameter names to their values
            - species_names : list
                List of species names in the order corresponding to the Jacobian matrix
        '''
        maker = cls(antimony_str)
        maker.build()
        return maker.jacobian_smat

    SymbolicJacobian = namedtuple("SymbolicJacobian", ["jacobian_smat", "b_smat"])
    def _makeReactionSymbolicJacobian(self, reaction: libsbml.Reaction)->SymbolicJacobian:
        """ Creates the symbolic A and B matrices for a given reaction.
        The B matrix is n X 1, and contains the rate at which the i-th species changes

        Args:
            reaction (libsbml.Reaction): _description_

        Returns:
            SymbolicJacobian
        """
        # Extract the rate constant and species for each reaction
        ##
        def findStrs(candidates: List[str], targets: List[str]) -> List[str]:
            # Find the candidates that are equal to the targets
            results = set(candidates).intersection(targets)
            return list(results)
        ##
        reaction_name = reaction.getId()
        # Get reactants (should be 0 or 1)
        num_reactants = reaction.getNumReactants()
        if num_reactants > 1:
            raise ValueError(f"Reaction {reaction_name} has {num_reactants} reactants. Only 0 or 1 allowed.")
        reactant_name = None
        if num_reactants == 1:
            reactant = reaction.getReactant(0)
            if reactant.getStoichiometry() != 1:
                raise ValueError(f"Reaction {reaction_name} has non-unit stoichiometry")
            reactant_name = reactant.getSpecies()
        # Parse the kinetic law to extract the rate constant, products, and their stoichiometries
        kinetic_law_str = reaction.getKineticLaw()
        kinetic_law_strs = kinetic_law_str.formula.split(" ").strip()
        kinetic_constants = findStrs(kinetic_law_strs, list(self.kinetic_constant_dct.keys()))
        if len(kinetic_constants) != 1:
            raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
        kinetic_constant_name = kinetic_constants[0]
        species_names = findStrs(kinetic_law_strs, self.species_names)
        if len(species_names) > 1:
            raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
        kinetic_species_name = species_names[0] if len(species_names) == 1 else None
        # Get products and their stoichiometries
        product_stoichiometry_dct: dict = {}  # key: product_id, value: stoichiometry
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            product_id = product.getSpecies()
            product_stoichiometry_dct[product_id] = product.getStoichiometry()
        # Construct the symbolic Jacobian for this reaction
        jacobian_smat = sp.Matrix.zeros(len(self.species_names), len(self.species_names))
        b_smat = sp.Matrix.zeros(len(self.species_names), 1)
        kinetic_constant_symbol = sp.Symbol(kinetic_constant_name)
        #   Handle the reactant
        if reactant_name is not None:
            reactant_idx = self._getSpeciesIndex(reactant_name)
            if kinetic_species_name is not None:
                kinetic_species_idx = self._getSpeciesIndex(kinetic_species_name)
                jacobian_smat[reactant_idx, kinetic_species_idx] = -kinetic_constant_symbol
            else:
                b_smat[reactant_idx] = -kinetic_constant_symbol
        #   Handle the products
        for product_id, stoich in product_stoichiometry_dct.items():
            product_idx = self._getSpeciesIndex(product_id)
            if reactant_name is not None:
                reactant_idx = self._getSpeciesIndex(reactant_name)
                jacobian_smat[product_idx, reactant_idx] = stoich * kinetic_constant_symbol
        #
        return self.SymbolicJacobian(jacobian_smat=jacobian_smat, b_smat=b_smat)