'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577 
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        S += 1e-34 # Prevent zero entries (Let's see how it goes)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        #print('NS',NS)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        #print('F nan?', F.isnan().any())

        G = (S + S[:, None])
        #print('S max?',S.abs().max())
        #print('S min?',S.abs().min())
        G.diagonal().fill_(np.inf)
        G = 1/G 
        #print('G nan? G zero? G max?', G.isnan().any(),G.min(),G.max())

        UdU = Ut @ dU
        VdV = Vt @ dV

        #print('Ut nan?',Ut.isnan().any())
        #print('Vt nan?',Vt.isnan().any())
        #print('dU nan?',dU.isnan().any())
        #print('dV nan?',dV.isnan().any())
        #print('UdU nan?',UdU.isnan().any())
        #print('VdV nan?',VdV.isnan().any())

        #print('UdU-UdU.t nan?',(UdU-UdU.t()).isnan().any())
        #print('VdV-VdV.t nan?',(VdV-VdV.t()).isnan().any())
        #print('F+G nan?',(F+G).isnan().any())
        #print('F-G nan?',(F-G).isnan().any())
        #print('(F+G)*(UdU-UdU.t()) nan?',((F+G)*(UdU-UdU.t())).isnan().any())
        #print('(F-G)*(VdV-VdV.t()) nan?',((F-G)*(VdV-VdV.t())).isnan().any())

        #print('F abs max?',F.abs().max())
        #print('G abs max?',G.abs().max())
        #print('F+G abs max?', (F+G).abs().max())
        #print('F+G abs min?', (F+G).abs().min())
        #print('(UdU-UdU.t()) abs max? ',(UdU-UdU.t()).abs().max())
        #print('(UdU-UdU.t()) abs min? ',(UdU-UdU.t()).abs().min())

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        #print('Su nan?',Su.isnan().any())
        #print('Sv nan?',Sv.isnan().any())

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #print('Get dA successfully! ',NS)
        return dA