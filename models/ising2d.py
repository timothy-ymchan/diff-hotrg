from .abstract_model import AbstractModel
import torch
from opt_einsum import contract
from scipy.integrate import quad
import numpy as np

class Ising2D(AbstractModel):
  N0 = 1
  name = "Ising2D"
  Tc = 2/np.log(1+np.sqrt(2))
  def __init__(self,temp:torch.Tensor):
    self.temp = temp
    self.device = torch.get_default_device() # So that the code is aware of the default device 
    self.Tc = torch.Tensor([Ising2D.Tc]).to(self.device)

  def get_params(self) -> dict:
    return {"T":self.temp,}

  def get_T(self)->torch.Tensor:
    T = self.temp
    # A nice paper that describe this construction is https://arxiv.org/pdf/1709.07460
    delta =torch.zeros([2,2,2,2])
    delta[0,0,0,0] = delta[1,1,1,1] = 1
    beta = 1/T
    sqrt_c,sqrt_s = torch.sqrt(torch.cosh(beta)), torch.sqrt(torch.sinh(beta))
    M = torch.zeros(2,2) # Have to do it this way to preserves gradient
    M[0,0] = M[1,0] = sqrt_c 
    M[0,1] = sqrt_s 
    M[1,1] = -sqrt_s
    return contract("abcd,ai,bj,ck,dl->ijkl",delta,M,M,M,M)

  def get_mag_op(self)->torch.Tensor:
    T = self.temp
    delta = torch.zeros([2,2,2,2])
    delta[0,0,0,0] = -1
    delta[1,1,1,1] = 1
    beta = 1/T
    sqrt_c,sqrt_s = torch.sqrt(torch.cosh(beta)), torch.sqrt(torch.sinh(beta))
    M = torch.Tensor([[sqrt_c,sqrt_s],[sqrt_c,-sqrt_s]]).to(self.device)
    return contract("abcd,ai,bj,ck,dl->ijkl",delta,M,M,M,M)

  def get_lnZ(self)->torch.Tensor:
    T = self.temp.to('cpu').detach().item() # as floating point number, and since we are using scipy, copy everything to cpu
    beta = 1/T
    k = 1/(np.sinh(2*beta))**2
    func = lambda theta: np.log(np.cosh(2*beta)**2 + (1/k)*np.sqrt(1+k**2-2*k*np.cos(theta)))
    return torch.Tensor([0.5*np.log(2) + (quad(func,0,np.pi))[0]/(2*np.pi)]).to(self.device)

  def get_mag(self)->torch.Tensor:
    temp = self.temp.detach() # Better not include this in computational graph
    if temp > self.Tc:
      return torch.zeros(1)
    beta = 1/temp
    K = 1/torch.sinh(2*beta)
    return (1-K**4)**(1/8)
    
  def get_Cv(self)->torch.Tensor:
    raise "Unimplemented"
    return torch.Tensor([0])

  def get_obs_th(self) -> dict:
    return {"mag":self.get_mag(),"Cv":self.get_Cv()}