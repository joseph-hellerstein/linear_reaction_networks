'''Container for control.TransferFunction and common methods.'''

import control # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Optional, List
import tellurium as te # type: ignore


AX = "ax"
FIGSIZE = "figsize"
INPUT = "input"
PREDICTION = "prediction"
SIMULATION = "simulation"
TIME = "time"
TIMES = list(np.linspace(0, 100, 1000))
TITLE = "title"


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
    
    def _simulate(self, model:str, times:Optional[List]=TIMES)->pd.DataFrame:
        """
        Simulates the model using tellurium

        Args:
            model (str): Antimony model
            times (Optional[np.ndarray], optional): Simulation times

        Returns:
            pd.DataFrame: columns
                TIME, INPUT, SIMULATION
        """
        rr = te.loada(model)
        selections = [TIME, self.input_name, self.output_name]
        rr.selections = selections
        data = rr.simulate(times[0], times[-1], len(times), selections=selections)   # type: ignore
        df = pd.DataFrame(data, columns=data.colnames)
        columns = [c[1:-1] if c[0] == '[' else c for c in df.columns]
        df[INPUT] = df[self.input_name]
        df[SIMULATION] = df[self.output_name]
        for column in columns:
            if not column in [TIME, INPUT, SIMULATION]:
                del df[column]
        return df

    def predict(self, data:Union[pd.DataFrame, str], times:Optional[List]=TIMES)->pd.DataFrame:
        """
        Using the transfer function to predict the output from the input.

        Args:
            data (pd.DataFrame or antimony model): Dataframe Columns
                TIME: times for predictions
                <input_name>
                <output_name> (simulated result)
            times (Optional[List], optional): Times for predictions

        Returns:
            pd.DataFrame: Columns
                TIME: times for predictions
                <input_name>
                SIMULATION: simulation of the output
                PREDICTION
        """
        if isinstance(data, str):
            df = self._simulate(data, times=times)
        else:
            df = data
        uvals = df[INPUT].values
        _, predictions = control.forced_response(self.transfer_function, T=df[TIME], U=uvals)
        df[PREDICTION] = predictions
        return df
    
    def verify(self, model:str, times:Optional[List]=TIMES, is_plot:bool=False, **kwargs)->bool:
        """
        Checks the predictions against the data.

        Args:
            model (str): Antimony model
            times (Optional[List], optional): Times for predictions
            kwargs (dict): Additional arguments for plotting

        Returns:
            bool
        """
        df = self.predict(model, times=times)
        simulations = df[SIMULATION].values
        predictions = df[PREDICTION].values
        # Check that the output is monotonic
        rmse = np.sqrt(np.sum(simulations - predictions)**2)   # type: ignore
        std = np.std(simulations)   # type: ignore
        if is_plot:
            if AX in kwargs.keys():
                ax = kwargs[AX]
            elif not FIGSIZE in kwargs.keys():
                kwargs[FIGSIZE] = [5,5]
                _, ax = plt.subplots(1, figsize=kwargs[FIGSIZE])
            ax.scatter(simulations, predictions, color="red", marker="*")
            ax.set_xlabel(SIMULATION)
            ax.set_ylabel(PREDICTION)
            max_simulated = np.max(simulations)   # type: ignore
            max_predictions = np.max(predictions)  # type: ignore
            max_value = max(max_simulated, max_predictions)
            ax.plot([0, max_value], [0, max_value], linestyle="--")
            if TITLE in kwargs.keys():
                title = kwargs[TITLE]
            else:
                title = "%s->%s" % (self.input_name, self.output_name)
            ax.set_title(title)
            plt.show()
        return rmse/std < 0.1