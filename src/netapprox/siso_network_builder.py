"""
Build a SISO reaction network from other networks.

:Author: Joseph L. Hellerstein
:Date: 2024-01-09
:Email: joseph.hellerstein@gmail.com
:License: MIT
"""

import numpy as np
import control  # type: ignore
import re
import tellurium as te  # type: ignore

IN = "_in"
OT = "_ot"
COMMENT_STR = "//"


class SISONetworkBuilder(object):

    def __init__(self):
        """
        """
        # Find the main module
        self.antimony = ""
        self.transfer_function = control.TransferFunction([1], [1])

    def copy(self):
        """
        Returns:
            AntimonyBuilder
        """
        raise NotImplementedError("Must implement")
    
    def __eq__(self, other):
        is_equal = True
        is_debug = False
        is_equal &= self.antimony == other.antimony
        if is_debug and (not is_equal):
            print("Failed 1")
        is_equal &= util.allEqual(self.antimony_strs, other.antimony_strs)
        if is_debug and (not is_equal):
            print("Failed 2")
        is_equal &= self._initialized_output == other._initialized_output
        if is_debug and (not is_equal):
            print("Failed 3")
        is_equal = is_equal and (all([self.symbol_dct[k] == other.symbol_dct[k] for k in self.symbol_dct.keys()]))
        if is_debug and (not is_equal):
            print("Failed 4")
        is_equal = is_equal and (all([self.symbol_dct[k] == other.symbol_dct[k] for k in other.symbol_dct.keys()]))
        if is_debug and (not is_equal):
            print("Failed 5")
        is_equal &= self.insert_pos == other.insert_pos
        if is_debug and (not is_equal):
            print("Failed 6")
        return is_equal

    def addStatement(self, statement:str):
        """
        Args:
            statement: str
        """
        def insert(stg, increment=1):
            self.antimony_strs.insert(self.insert_pos, stg)
            self.insert_pos += increment
        #
        insert(statement)

    def makeComment(self, comment:str):
        """
        Args:
            comment: str
        """
        self.addStatement("%s %s" % (COMMENT_STR, comment))