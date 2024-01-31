'''Container for control.TransferFunction and common methods.'''

import control # type: ignore
import pandas as pd
import numpy as np
from typing import Union, Optional, List
import tellurium as te # type: ignore


TIME = "time"
TIMES = list(np.linspace(0, 100, 1000))


class NamedTransferFunction(object):

    def __init__(self, input_name:str, output_name:str, transfer_function:control.TransferFunction):
        """

        Args:
            input_name (str): name of the input variable
            output_name (str): name of the output variable
            transfer_function (control.TransferFunction)
        """
        self.input_name = input_name
        self.output_name = output_name
        self.transfer_function = transfer_function

    def __repr__(self):
        return f"NamedTransferFunction({self.input_name}, {self.output_name}, {self.transfer_function})"
    
    def _simulate(self, model:str, times:Optional[List]=TIMES, selections:Optional[List[str]]=None)->pd.DataFrame:
        """
        Simulates the model using tellurium

        Args:
            model (str): Antimony model
            times (Optional[np.ndarray], optional): Simulation times
            selections (Optional[List[str]], optional): Variables to select

        Returns:
            pd.DataFrame
        """
        rr = te.loada(model)
        if selections is None:
            selections = [self.input_name, self.output_name]
        if not TIME in selections:
            selections = [TIME] + selections
        rr.selections = selections
        data = rr.simulate(times[0], times[-1], len(times), selections=selections)   # type: ignore
        df = pd.DataFrame(data, columns=data.colnames)
        columns = [c[1:-1] if c[0] == '[' else c for c in df.columns]
        df.columns = columns # type: ignore
        return df

    def predict(self, data:Union[pd.DataFrame, str], times:Optional[List]=TIMES)->pd.DataFrame:
        """
        Using the transfer function to predict the output from the input.

        Args:
            data (pd.DataFrame or antimony model): Dataframe Columns
                TIME: times for predictions
                <input_name>

        Returns:
            pd.DataFrame: Columns
                TIME: times for predictions
                <input_name>
                <output_name>
        """
        if isinstance(data, str):
            df = self._simulate(data, times=times, selections=[self.input_name])
        else:
            df = data
        uvals = df[self.input_name].values
        _, predictions = control.forced_response(self.transfer_function, T=df[TIME], U=uvals)
        df[self.output_name] = predictions
        return df
    
    def verify(self, data:Union[pd.DataFrame, str], is_plot:bool=False)->bool:
        """
        Checks the predictions against the data.

            data (pd.DataFrame or antimony model): Dataframe Columns
                TIME: times for predictions
                <input_name>

        Returns:
            bool
        """
        raise NotImplementedError("Must implement")
        return False