import torch
from opt_einsum import contract

class HOTRG:
  def __init__(self,model,chi_max,renormalize_obs=True):
    self.model = model
    self.T = [self.model.get_T()]
    self.obs_ss = self.model.get_single_site_obs() # Get single site observables
    self._traceT = [contract("urur",self.T[-1])]
    self._maxT = []
    self._maxObs = {name:[] for name in self.obs_ss.keys()}
    self._isom_row = [] # Isometries along rows 
    self._isom_col = [] # ISoemtries along cols
    self.iter = 0
    self.chi_max = chi_max
    self.Nsizes = [model.N0]
    self.scale = 4

  def run(self,niter,verbose=True): # Run for n more iterations
    for i in range(niter):
      self.iter += 1
      self._renormalize()

      #if verbose and i% 5 == 0:
      #  lnZ_true = self.model.get_lnZ_theory()
      #  lnZ_RG = self.lnZ()
      #  print("Steps: ", i,"System size",f"{self.Nsizes[-1]:.3e}","lnZ_RG ", lnZ_RG, " Error: ",np.abs((lnZ_RG-lnZ_true)/lnZ_true))
      #  for name in self.obs_ss.keys():
      #    print(f"Observable {name}: RG", self.observable(name), " Theory", self.model.get_mag_theory(),"\n")

  def lnZ(self,iter=-1,per_site=True): # Get lnZ of the n iteration
    assert iter <= self.iter, "iter out of range"
    if iter == -1:
      iter = self.iter
    lnZ = 0
    for i in range(iter):
      lnZ = self.scale*lnZ + torch.log(self._maxT[i])
    lnZ += torch.log(self._traceT[iter])
    if per_site:
      for i in range(iter):
         lnZ /= self.scale
      return lnZ
      #return lnZ/self.Nsizes[iter] # Free energy per spin
    return lnZ

  def observable(self,name,iter=-1):
    assert name in self.obs_ss.keys(), f"Observable {name} not found"
    assert iter <= self.iter, "iter out of range"
    if iter == -1:
      iter = self.iter
    return contract("urur",self.obs_ss[name][iter]) /contract("urur",self.T[iter])

  def _renormalize(self):
    T = self.T[-1]
    chi_max = self.chi_max

    # Do one renormalization steps. Modify the object in-place
    chi_u,chi_r,chi_d,chi_l = T.shape

    # Merge legs in X-direction
    # u1 r1 d1 l1 *delta(d1,u2) * u2 r2 d2 l2 -> u1 (r1 r2) d2 (l1 l2)
    T_merged_x = contract("ijkl,knop->ijnolp",T,T)
    T_merged_x = T_merged_x.reshape((chi_u,chi_r*chi_r,chi_d,chi_l*chi_l))
    RRx = self._get_X_rank_reducer(T_merged_x,chi_max) # (chi_max, chi^2)

    # Rank reduction along x
    T_rrx = contract("ar,bl,urdl->uadb",RRx,RRx,T_merged_x)

    # Merge legs in Y-direction
    chi_u,chi_r,chi_d,chi_l = T_rrx.shape
    T_merged_Y = contract("ijkl,mnoj->imnkol",T_rrx,T_rrx)
    T_merged_Y = T_merged_Y.reshape(chi_u*chi_u,chi_r,chi_d*chi_d,chi_l)
    RRy = self._get_Y_rank_reducer(T_merged_Y,chi_max)

    T_new = contract("au,bd,urdl->arbl",RRy,RRy,T_merged_Y)

    # Update register
    T_max = torch.max(torch.abs(T_new))
    self.T.append(T_new/T_max)
    self.Nsizes.append(self.scale*self.Nsizes[-1])
    self._maxT.append(T_max)
    self._traceT.append(contract("urur",T_new))

    # Renormalize measurement operators and update register
    # Impurity averaging (https://www.sciencedirect.com/science/article/pii/S001046551830362X)

    for name in self.obs_ss.keys():
      To = self.obs_ss[name][-1]

      # Get the up block
      To_up = self._apply_trunc_x(To,T,RRx)
      # Get the down block
      To_dn = self._apply_trunc_x(T,To,RRx)

      # Renormalize vertically
      To_y = 0.5*(To_up + To_dn)

      # Get left and right block
      To_lft = self._apply_trunc_y(To_y,T_rrx,RRy)
      To_rgt = self._apply_trunc_y(T_rrx,To_y,RRy)

      # Renormalize horizontally
      To_new = 0.5*(To_lft + To_rgt)

      #self._maxObs[name].append(To_new_max)
      self.obs_ss[name].append(To_new/T_max) # Normalize by the same factor as the free energy

  def _apply_trunc_x(self,up,down,proj_x):
      # Apply projective truncation on up down
      chi_u, chi_r, chi_d, chi_l = up.shape
      blk = contract("ijkl,knop->ijnolp",up,down)
      blk = blk.reshape((chi_u,chi_r*chi_r,chi_d,chi_l*chi_l))
      return contract("ar,bl,urdl->uadb",proj_x,proj_x,blk)

  def _apply_trunc_y(self,right,left,proj_y):
      # Apply projective truncation on left and right
      chi_u, chi_r, chi_d, chi_l = right.shape
      blk = contract("ijkl,mnoj->imnkol",right,left)
      blk = blk.reshape((chi_u*chi_u,chi_r,chi_d*chi_d,chi_l))
      return contract("au,bd,urdl->arbl",proj_y,proj_y,blk)


  def _get_X_rank_reducer(self,T,chi_max):
      # Input: 4 legged tensor T
      _, chi_r, _, chi_l = T.shape
      assert chi_r == chi_l, f"The bond dimensions of the left bond ({chi_l}) and right bond ({chi_r}) must be equal"

      # Assume symmetry and just do one SVD
      ML = T.transpose([3,0,1,2]).reshape(chi_l,-1)
      ML = contract("ij,kj->ik",ML,ML)
      UL,_,_ = mysvd(ML,chi_max)
      del ML
      return np.transpose(UL)

      # Remark: If the tensors are not symmetric, we will need to do left-right comparison in truncation

  def _get_Y_rank_reducer(self,T,chi_max):
      # T (u r d l)
      chi_u, _, chi_d, _ = T.shape
      assert chi_u == chi_d, f"The bond dimensions of the up bond ({chi_u}) and down bond ({chi_d}) must be equal"
      return self._get_X_rank_reducer(T.transpose([1,2,3,0]),chi_max)
