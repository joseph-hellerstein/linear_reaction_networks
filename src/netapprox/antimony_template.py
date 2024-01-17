"""
Antimony template extends the Antimony modeling language to supports template variables.
  <<model_name>>: Name of the model
  <<submodel_name__n>>: Name of the n-th submodel model
Features:
  1. Convert an existing model to a template
  2. Expand a template to an antimony model

:Author: Joseph L. Hellerstein
:Date: 2024-01-09
:Email: joseph.hellerstein@gmail.com
:License: MIT
"""

import netapprox.constants as cn

import re
import tellurium as te # type: ignore
from typing import Optional, List

DEFAULT_MODEL_NAME = "main_model"


class AntimonyTemplate():

    def __init__(self, antimony: str):
        """
        Args:
            antimony: Antimony string
        """
        # Find the main module
        self.original_antimony = antimony
        self.substituted_antimony = ""
        self.model_name = ""
        self.initialize()   # Sets self.substituted_antimony and self.model_name

    def initialize(self):
        """
        Initializes the antimony as a template model
        """
        self.substituted_antimony = self.original_antimony
        substituted_antimony = self.original_antimony
        model_name = self._findMainModelName()
        if len(model_name) == 0:
            self.makeModularModel()
            model_name = DEFAULT_MODEL_NAME
        else:
            new_model_name = "*%s" % model_name
            substituted_antimony = substituted_antimony.replace(new_model_name, cn.TE_MODEL_NAME)
            self.substituted_antimony = substituted_antimony
        self.model_name = model_name

    def _extractModelName(self, line: str)->str:
        # Extracts the name of the model from the line
        start_pos = line.find("*") + 1
        end_pos = line.find("(")
        if (start_pos < 0) or (end_pos < 0) or (end_pos < start_pos):
            raise RuntimeError("Unable to extract model name from line: %s" % line)
        name = line[start_pos:end_pos]
        name = name.strip()
        return name

    def _findMainModelName(self)->str:
        """
        Finds the position of the end statement for the main model. An empty
        string is returned if no model is found

        Returns:
            str (model name)
        """
        NULL_MODEL_NAME = ""
        # Finds the name of the top level model
        antimony_strs: List[str] = self.original_antimony.split("\n")
        main_model_line = ""
        name_pos = -1
        for line in antimony_strs:
            result = re.search("model .*[*].*()", line)
            if result:
                main_model_line = line
                break
        if len(main_model_line) == 0:
            return NULL_MODEL_NAME
        return self._extractModelName(main_model_line)
    
    def copy(self)->"AntimonyTemplate":
        """
        Returns:
            SISOAntimonyBuilder
        """
        template = AntimonyTemplate(self.original_antimony)
        template.substituted_antimony = self.substituted_antimony
        return template
    
    def __repr__(self)->str:
        return self.substituted_antimony
    
    def __eq__(self, other: "AntimonyTemplate")->bool:  # type: ignore
        is_equal = True
        is_debug = False
        is_equal &= self.original_antimony == other.original_antimony
        if is_debug and (not is_equal):
            print("Failed 1")
        is_equal &= self.substituted_antimony == other.substituted_antimony
        if is_debug and (not is_equal):
            print("Failed 2")
        return is_equal
    
    def setTemplateVariable(self, var_name: str, value: str)->None:
        """
        Substitutes all occurrences of the template variable for the string.

        Args:
            var_name (str): template variable with angle brackets ('<<', '>>')
            value (str): value to substitute
        """
        if not "<<" in var_name:
            raise ValueError("Template variable must be enclosed in angle brackets: %s" % var_name)
        if self.substituted_antimony.count(var_name) == 0:
            raise ValueError("Template variable not found in antimony: %s" % var_name)
        self.substituted_antimony = self.substituted_antimony.replace(var_name, value)

    def makeModularModel(self)->None:
        """
        Transforms the antimony string into a modular model.
        """
        lines = self.substituted_antimony.split("\n")
        for line in lines:
            result = re.search("model .*[*].*()", line)
            if result:
                return
        suffix = "\nend"
        self.substituted_antimony = "model %s()\n" % cn.TE_MODEL_NAME + self.substituted_antimony + suffix

    def isValidAntimony(self)->bool:
        """
        Returns:
            bool: True if the antimony is valid
        """
        if "<<" in self.substituted_antimony:
            return False
        try:
            te.loada(self.substituted_antimony)
            return True
        except Exception as e:
            print("Error: %s" % e)
            return False
    
    @staticmethod
    def makeSubmodelTemplateName(idx: int)->str:
        """
        Args:
            idx: index of the submodel
        Returns:
            str: name of the template submodel
        """
        return cn.TE_SUB_MODEL_NAME % idx