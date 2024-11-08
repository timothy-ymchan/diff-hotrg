from abc import ABC
import torch

class AbstractModel(ABC):
    N0:int = None
    name:str = None
    def __init__(self,params:dict):
        pass

    def get_T(self)->torch.Tensor:
        pass
    
    def get_lnZ(self)->torch.Tensor:
        pass 

    def get_params(self)->dict:
        pass
    
    def get_obs_th(self)->dict:
        pass