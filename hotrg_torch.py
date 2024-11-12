import torch
from opt_einsum import contract
from models import AbstractModel
from svd import SVD


svd = SVD.apply

def merge_tensor(T1:torch.Tensor,T2:torch.Tensor,direction:str)->torch.Tensor:
  # Assume input tensor of the form urdl (efgh), assume bo
  chi1_u, chi1_r, chi1_d, chi1_l = T1.size()
  chi2_u, chi2_r, chi2_d, chi2_l = T2.size()
  #print(T1.size(),T2.size())
  along = {"X":"urdl,efgr->uefdgl","Y":"urdl,dfgh->urfglh"}
  Tnew:torch.Tensor = contract(along[direction],T1,T2) # Stack and contract tensor along direction
  #print(Tnew.size())
  new_axis = {"X":(chi1_u*chi2_u,chi1_r,chi1_d*chi2_d,chi1_l),"Y":(chi1_u,chi1_r*chi2_r,chi1_d,chi1_l*chi2_l)}
  #print(new_axis[direction])
  return Tnew.reshape(new_axis[direction])


def get_isometry(T:torch.Tensor, merge_direction:str, max_dim:int ) -> torch.Tensor:
  # If the tensors are merged along X, isom should be applied on Y, vice versa
  along = {"X":"urdl,erdl->ue","Y":"urdl,urdh->lh"} 
  M:torch.Tensor = contract(along[merge_direction],T,torch.conj(T))
  #print('M',M.size())
  U,S,Vh = svd(M) # U.shape = (bond dim, svd rank)
  isom = (U[:,:max_dim].T)
  return isom


class HOTRG:
  def __init__(self,model:AbstractModel,chi_max:int,disable_ckpt=False):
    self.model = model
    self.T = [self.model.get_T()]
    self._traceT:list[torch.Tensor] = [contract("urur",self.T[-1])]
    self._maxT:list[torch.Tensor] = []
    self._isoms:list[torch.Tensor] = []

    self.iter = 0
    self.chi_max = chi_max
    self.Nsizes = [model.N0]
    self.scale = 2
    self.sweep_cycle = ["X","Y"]

    # Construct default checkpoint name
    self.default_ckpt = self.get_default_ckpt()
    self.disable_ckpt = disable_ckpt

    self.n_ckpt = 5 # Save checkpoint every 5 iterations
    self.n_log = 1

    self.device = torch.get_default_device()
    
    
  def run(self,niter:int,verbose=True):
    self.n_log = niter // 10 if niter //10 > 1 else 1
    print(f"Renormalizing {self.model.name}")
    print(f"Current lattice size: {self.Nsizes[-1]}\tCurrent iterations: {self.iter}")
    if self.disable_ckpt:
      print('Warning: Checkpoint is disabled, nothing will be saved')
    else:
      print(f"Saving checkpoints every {self.n_ckpt} steps")
      print(f"Checkpoint will in saved at {self.default_ckpt}")
    if verbose:
      print(f"Printing logs every {self.n_log} steps")
    


    for i in range(niter):
      self._renormalize()

      if i % self.n_ckpt == 0:
        self.save_checkpoint()
      if verbose and i % self.n_log == 0:
        lnZ_RG = self.get_lnZ()
        lnZ_Th = self.model.get_lnZ()
        print('Iteration: ', i, f'Size {self.Nsizes[-1]:.3e}','RG lnZ: ',lnZ_RG.item(),'Theory lnZ: ',lnZ_Th.item(), f'Error: {(torch.abs(lnZ_Th-lnZ_RG)/lnZ_RG).item():.4e}')
    
    print('Completed ',niter,' iterations of renormalization')
    print(f'Saving output to {self.default_ckpt}...')
    self.save_checkpoint()

  def get_lnZ(self,iter=-1,per_site=True,detach=False)->torch.Tensor: # Get lnZ of the n iteration
    assert iter <= self.iter, "iter out of range"
    if iter == -1:
      iter = self.iter
    lnZ = torch.log(self._traceT[iter])
    for i in range(iter):
      lnZ = (lnZ + torch.log(self._maxT[iter-1-i]))/self.scale
    lnZ /= self.Nsizes[0]
    return lnZ
  
    """
    # Probably too unstable numerically. Better to compute per site lnZ directly
    for i in range(iter):
      lnZ = self.scale*lnZ + torch.log(self._maxT[i])
    lnZ += torch.log(self._traceT[iter])
    if per_site:
      size = torch.Tensor([self.Nsizes[iter]]).to(self.device)
      lnZ = lnZ/size # Free energy per spin
    return lnZ
    """
  
  def _renormalize(self):
    T = self.T[-1]
    chi_max = self.chi_max
    direction = self.sweep_cycle[self.iter % len(self.sweep_cycle)]
    TT = merge_tensor(T,T,direction)
    #print(TT.size())
    isom = get_isometry(TT,direction,self.chi_max) # legs (chi_new, chi0)

    # Apply truncation
    trunc_along = {"X":"ua,arbl,db->urdl","Y":"la,ubda,rb->urdl"} # direction means merge direction
    T_new:torch.Tensor = contract(trunc_along[direction],isom,TT,torch.conj(isom))
    #print(T_new.size())

    # Rescaling
    T_max = torch.max(torch.abs(T_new))
    self.T.append(T_new/T_max)
    self.Nsizes.append(self.scale*self.Nsizes[-1])
    self._maxT.append(T_max)
    self._traceT.append(contract("urur",T_new))
    self._isoms.append(isom)

    # Update iter
    self.iter += 1

  def save_checkpoint(self,path:str=None):
    if self.disable_ckpt:
      return
    path = self.default_ckpt if path is None else path
    state_dict = {
      "model_params":self.model.get_params(),
      "chi":self.chi_max,
      "iter":self.iter,
      "Nsizes":self.Nsizes,
      "T":self.T,
      "isom":self._isoms,
      "maxT":self._maxT
    }
    torch.save(state_dict,path)

  def get_default_ckpt(self):
    path = f"./{self.model.name}-chi{self.chi_max}"
    for param,val in self.model.get_params().items():
      val = val.detach().to('cpu').item()
      path += f'-{param}_{val:.5f}'
    path += '.pt'
    return path