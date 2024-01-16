from typing import Generic, TypeVar, List, Optional, Any
import control # type: ignore
import numpy as np
T = TypeVar('T')


class SISONetwork(Generic[T]):

    def __init__(self, input_name:str, output_name:str, model_reference:str, kI:float, kO:float,
                            transfer_function:control.TransferFunction, operating_region: List[float]) -> None: ...

    def plotStaircaseResponse(self, initial_value:Optional[float]=...,
                            final_value:Optional[float]=..., num_step:Optional[float]=...,
                              **kwargs)-> None: ...

    @classmethod
    def makeTwoSpeciesNetwork(cls, kI:float, kO:float,
                              operating_region: Any)->"SISONetwork": ...
    
    @classmethod
    def makeCascade(cls, input_name:str, output_name:str, kIs:List[float], kOs:List[float],
                    operating_region: Any)->"SISONetwork": ...
    
    def concatenate(self, other:"SISONetwork")->"SISONetwork": ...
    
    def forkjoin(self, other:"SISONetwork")->"SISONetwork": ...
    
    def loop(self, k1:float, k2:float, k3:float, k4:float, k5:float, k6:float)->"SISONetwork": ...
    
    def amplify(self, k1, k2)->"SISONetwork": ...