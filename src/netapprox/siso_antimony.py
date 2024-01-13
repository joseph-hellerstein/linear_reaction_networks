"""
Handles manipulation of antimony strings in support of network transformations.

:Author: Joseph L. Hellerstein
:Date: 2024-01-09
:Email: joseph.hellerstein@gmail.com
:License: MIT
"""

from netapprox.siso_network import SISONetwork

import numpy as np
import control  # type: ignore
import re
import tellurium as te  # type: ignore
from typing import Optional, List

IN = "_in"
OT = "_ot"
COMMENT_STR = "//"
MODEL_NAME = "main_model"

class SISOAntimony(object):

    def __init__(self, antimony: str):
        """
        Args:
            antimony: Antimony string
        """
        # Find the main module
        self.antimony_strs: List[str] = antimony.split("\n")
        self.main_model_name, self.name_pos = self._findMainModelName()

    def _extractModelName(self, line: str)->str:
        # Extracts the name of the model from the line
        start_pos = line.find("*") + 1
        end_pos = line.find("(")
        if (start_pos < 0) or (end_pos < 0) or (end_pos < start_pos):
            raise RuntimeError("Unable to extract model name from line: %s" % line)
        return line[start_pos:end_pos]

    def _findMainModelName(self)->tuple[str, int]:
        """
        Finds the position of the end statement for the main model.

        Returns:
            str (model name)
            int (position in list of strings)
        """
        # Finds the name of the top level model
        main_model_line = ""
        name_pos = -1
        for pos, line in enumerate(self.antimony_strs):
            result = re.search("model .*[*].*()", line)
            if result:
                main_model_line = line
                name_pos = pos
                break
        if len(main_model_line) == 0:
            raise ValueError("Could not find a main model!")
        return self._extractModelName(main_model_line), name_pos
    
    def copy(self):
        """
        Returns:
            SISOAntimonyBuilder
        """
        return SISOAntimony("\n".join(self.antimony))
    
    def __repr__(self):
        return self.getAntimony()
    
    def getAntimony(self, mode_name=None, is_main=False)->str:
        """
        Returns:
            Antimony string
        """
        if mode_name is None:
            mode_name = self.main_model_name
        if is_main:
            star = "*"
        else:
            star = ""
        new_antimony = self.copy()
        new_antimony.antimony_strs[self.name_pos] = "model %s%s()" % (star, mode_name)
        return "\n".join(new_antimony.antimony_strs)
    
    def __eq__(self, other: "SISOAntimony"):  # type: ignore
        is_equal = True
        is_debug = False
        is_equal &= self.antimony_strs == other.antimony_strs
        if is_debug and (not is_equal):
            print("Failed 1")
        return is_equal
    
    def toggleMain(self, is_main=False)->"SISOAntimony":
        """
        Toggles if main model.
        Args:
            is_main: True if main model
        """
        antimony = self.copy()
        if is_main:
            star = "*"
        else:
            star = ""
        antimony.antimony_strs[self.name_pos] = "model %s%s()" % (star, antimony.main_model_name)
        return antimony
    
    def append(self, other: "SISOAntimony")->"SISOAntimony":
        """
        Creates an new SISOAntimony by appending the other to self.
        Args:
            other: SISOAntimony
        """
        antimony = self.copy()
        antimony.antimony_strs.append["\n"]
        antimony.antimony_strs.extend(other.antimony_strs)
        return antimony