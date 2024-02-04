'''Container for control.TransferFunction and common methods. Extension to MISO transfer functions.'''

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
    def __init__(self, input_name:Union[str, List[str]], output_name:str,
                 transfer_function:Union[control.TransferFunction, List[control.TransferFunction]]):
        """

        Args:
            input_name (str, list-str): name(s) of the input variable
            output_name (str): name of the output variable
            transfer_function (control.TransferFunction, list-control.TransferFunction): transfer function(s
        """
        if isinstance(input_name, str):
            input_name = [input_name]
        self.input_names = input_name
        self.output_name = output_name
        if isinstance(transfer_function, control.TransferFunction):
            transfer_function = [transfer_function]
        self.transfer_functions = transfer_function

    def __repr__(self):
        stg = "%s = " % self.output_name
        terms = []
        for idx, input_name in enumerate(self.input_names):
            tf_stg = str(self.transfer_functions[idx])
            terms.append(f"{tf_stg}{input_name}")
        stg += " + ".join(terms)
        return stg
    
    def __eq__(self, other:object) -> bool:
        if isinstance(other, NamedTransferFunction):
            return self.input_names == other.input_names  \
                and self.output_name == other.output_name and self.transfer_functions == other.transfer_functions
        return False
    
    def copy(self):
        return NamedTransferFunction(self.input_names, self.output_name, self.transfer_functions)
    
    def _getInputColumn(self, input_name:str)->str:
        return "%s__%s" % (INPUT, input_name)
    
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
        selections = [TIME, self.output_name]
        selections.extend(self.input_names)
        rr.selections = selections
        data = rr.simulate(times[0], times[-1], len(times), selections=selections)   # type: ignore
        df = pd.DataFrame(data, columns=data.colnames)
        for input_name in self.input_names:
            column = self._getInputColumn(input_name)
            df[column] = df[input_name]
            del df[input_name]
        df[SIMULATION] = df[self.output_name]
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
        miso_predictions = np.zeros(len(df.index))
        for idx, transfer_function in enumerate(self.transfer_functions):
            column = self._getInputColumn(self.input_names[idx])
            _, predictions = control.forced_response(transfer_function, T=df[TIME], U=df[column].values)
            miso_predictions += predictions
        df[PREDICTION] = miso_predictions
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

    def evaluate(self, model:str, times:Optional[List]=TIMES, fractional_deviation:float=0.01,
                 **kwargs)->Tuple[pd.DataFrame, float]:
        """
        Checks the predictions against the data.

        Args:
            model (str): Antimony model
            times (Optional[List], optional): Times for predictions
            fractional_deviation (float, optional): Fractional deviation for the score (default=0.01)
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
        if len(errs) == 0:
            errs = np.array([np.isclose(s, p) for s, p in zip(simulations, predictions)])
        score = np.sum(errs <= fractional_deviation)/len(errs)
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
                reactant_str = "+".join(self.input_names)
                title = "%s->%s, score (fdev): %1.2f (%1.2f)" % (reactant_str, self.output_name, score, fractional_deviation)
            ax.set_title(title)
            plt.show()
        return df, float(score)