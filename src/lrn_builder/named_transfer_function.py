'''Container for control.TransferFunction and common methods.'''

import control # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
import tellurium as te # type: ignore


AX = "ax"
FIGSIZE = "figsize"
INPUT = "input"
IS_PLOT = "is_plot"
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
    
    def __eq__(self, other:object) -> bool:
        if isinstance(other, NamedTransferFunction):
            return self.input_name == other.input_name  \
                and self.output_name == other.output_name and self.transfer_function == other.transfer_function
        return False
    
    def copy(self):
        return NamedTransferFunction(self.input_name, self.output_name, self.transfer_function)
    
    def simulate(self, model:str, times:Optional[List]=TIMES)->pd.DataFrame:
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
            df = self.simulate(data, times=times)
        else:
            df = data
        uvals = df[INPUT].values
        _, predictions = control.forced_response(self.transfer_function, T=df[TIME], U=uvals)
        df[PREDICTION] = predictions
        return df

    def score(self, model:str, times:Optional[List]=TIMES, **kwargs)->float:
        """
        Scores the accuracy of the transfer function based on the fraction of step predictions < 0.01.

        Args:
            model (str): Antimony model
            times (Optional[List], optional): Times for predictions
            kwargs (dict):
                is_plot: bool (Plot the results)
                initial_value: float (Initial value for the staircase)
                final_value: float (Initial value for the staircase)
                num_step: int (Initial value for the staircase)

        Returns:
            float (0 <= score <= 1)
        """
        _, score = self.evaluate(model, times=times, **kwargs)
        return score

    def evaluate(self, model:str, times:Optional[List]=TIMES, frc_dev=0.01, **kwargs)->Tuple[pd.DataFrame, float]:
        """
        Checks the predictions against the data.

        Args:
            model (str): Antimony model
            times (Optional[List], optional): Times for predictions
            kwargs (dict): Additional arguments for plotting

        Returns:
            pd.DataFrame: Columns
                TIME: times for predictions
                INPUT
                SIMULATION: simulation of the output
                PREDICTION
        """
        if not IS_PLOT in kwargs.keys():
            kwargs[IS_PLOT] = False
        is_plot = kwargs[IS_PLOT]
        #
        df = self.predict(model, times=times)
        simulations = df[SIMULATION].values
        predictions = df[PREDICTION].values
        # Check that the output is monotonic
        errs = np.array([np.abs(s - p)/s for s, p in zip(simulations, predictions) if not np.isclose(s, 0)])
        score = np.sum(errs <= frc_dev)/len(errs)
        if is_plot:
            if AX in kwargs.keys():
                ax = kwargs[AX]
            else:
                if not FIGSIZE in kwargs.keys():
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
        return df, score