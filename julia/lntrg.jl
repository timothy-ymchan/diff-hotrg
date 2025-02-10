using TensorKit 
using TensorOperations 
include("./rg.jl")
include("trotter_sukuzi.jl")
include("./looptrg.jl")

mutable struct LNTRG_layer <: RG_layer
    
end

function LNTRG_next(T::TensorMap,χ::Int)
    # Assumes T[i,j,k,l] : V⊗V←V⊗V 
    U,D = SVD_factorize(T,(1,2),(3,4),truncdim(χ))
    L,R = SVD_factorize(T,(1,3),(2,4),truncdim(χ))
    @tensor T_new[i,j,k,l] := D[i,kk,ll]*L[ll,jj,k]*U[ii,jj,l]*R[j,kk,ii]
    return T_new
end

function EFTRG_next(Ta::TensorMap,Tb::TensorMap,χ::Int)
    # The original entanglement filtering algorithm in Gu-Wen 2009
    U,D = SVD_factorize(Tb,(1,2),(3,4),truncdim(χ))
    L,R = SVD_factorize(Ta,(1,3),(2,4),truncdim(χ))
    # Ta,Tb = entanglement_filtering(Ta,Tb)
    # Ta,Tb = LNTRG_TwoSite(Ta,Tb,χ)
    # return Ta,Tb
end

function LNTRG_TwoSite(Ta::TensorMap,Tb::TensorMap,χ::Int)
    # Assumes T[i,j,k,l] : V⊗V←V⊗V 
    U,D = SVD_factorize(Ta,(1,2),(3,4),truncdim(χ))
    L,R = SVD_factorize(Tb,(1,3),(2,4),truncdim(χ))
    @tensor T_new[i,j,k,l] := D[i,kk,ll]*L[ll,jj,k]*U[ii,jj,l]*R[j,kk,ii]
    return T_new,T_new # I think in one step of TRG we basically merges two sites
end


function LNTRG_renormalize(T::TensorMap,χ::Int,steps::Int)
    for i in range(1,steps)
        T = LNTRG_next(T,χ)
        T /= norm(T)
    end
    return T
end

function EFTRG_renormalize(Ta::TensorMap,Tb::TensorMap,χ::Int,steps::Int)
    for i in range(1,steps)
        Ta,Tb = EFTRG_next(Ta,Tb,χ)
        Ta /= norm(Ta)
        Tb /= norm(Tb)
    end
    return Ta,Tb
end
