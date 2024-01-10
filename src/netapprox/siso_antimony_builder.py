"""
Builds a SISO reaction networks.

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

class SISOAntimonyBuilder(object):

    def __init__(self, antimony: Optional[str]=None):
        """
        Args:
            antimony: Antimony string
        """
        # Find the main module
        if antimony is None:
            antimony = ""
        self.antimony_strs = antimony.split("\n")
        # Find the "model" and "end" statements
        self.start_pos, self.end_pos = self._initializeInsertPosition()
        # Find the main module
        rr = te.loada(antimony)
    
    def copy(self):
        """
        Returns:
            SISOAntimonyBuilder
        """
        builder = SISOAntimonyBuilder("\n".join(self.antimony))
        return builder
    
    def __eq__(self, other: "SISOAntimonyBuilder"):  # type: ignore
        is_equal = True
        is_debug = False
        is_equal &= self.antimony_strs == other.antimony_strs
        if is_debug and (not is_equal):
            print("Failed 1")
        return is_equal
    
    def _extractModelName(self, line):
        # Extracts the name of the model from the line
        start_pos = line.find("*") + 1
        end_pos = line.find("(")
        if (start_pos < 0) or (end_pos < 0) or (end_pos < start_pos):
            raise RuntimeError("Unable to extract model name from line: %s" % line)
        return line[start_pos:end_pos]

    def _initializeInsertPosition(self)->List[int]:
        """
        Finds the position of the end statement for the main model. If there is no model, then one is created.

        Returns:
            start_position of model
            end_position of model
        """
        # Handle empty model
        if len(self.antimony_strs) == 0:
            self.antimony_strs = ["model *%s()" % MODEL_NAME, "end"]
            return [0, 1]
        # Finds the name of the top level model
        last_model_start = -1
        for pos, line in enumerate(self.antimony_strs):
            result = re.search("model .*[*].*()", line)
            if result:
                last_model_start = pos
        if last_model_start < 0:
            raise ValueError("Could not find a main model!")
        # Finds the end of the top level model
        for pos, line in enumerate(self.antimony_strs[last_model_start:]):
            new_line = line.strip()
            if new_line == "end":
                return [last_model_start, pos + last_model_start]
        raise ValueError("Could not find end of main model!")

    def __repr__(self):
        return "\n".join([str(o) for o in self.antimony_strs])

    def addStatement(self, statement):
        """
        Args:
            statement: str
        """
        self.antimony_strs.insert(self.end_pos, statement)
        self.end_pos += 1

    def makeComment(self, comment):
        """
        Args:
            comment: str
        """
        self.addStatement("%s %s" % (COMMENT_STR, comment))

    def appendModel(self, other: "SISOAntimonyBuilder", comment: str=""):
        """
        Args:
            other: SISOAntimonyBuilder
        """
        full_comment = "//VVVVVVVVV %s VVVVVVVVV" % comment
        self.makeComment(full_comment)
        full_comment = "//^^^^^^^^^^^^^^^"
        self.makeComment(full_comment)
        # Add the other model
        for pos in range(other.start_pos+1, other.end_pos-1):
            self.addStatement(other.antimony_strs[pos])
        # Add anything that follows the model definition
        for pos in range(other.end_pos, len(other.antimony_strs)):
            self.addStatement(other.antimony_strs[pos])