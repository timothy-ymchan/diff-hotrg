using TensorKit
using TensorOperations
include("./rg.jl")

# ==========================================
#                HOTRG 2D
# ==========================================

# Tensor shape 
#                             ↓ 4/i 
# T[1/l,2/k,3/j,4/i] = 1/l ← (T) ← 3/j
#                         2/k ↓

mutable struct HOTRG2D_layer <: RG_layer 
    T::TensorMap
    norm::Real
    along::Symbol
    U::TensorMap # Rank reducer to be applied along `along`

    HOTRG2D_layer(T::TensorMap,norm::Real,along::Symbol,U::TensorMap) = new(T,norm,along,U)
end

function next_layer(layers::RG_layers{HOTRG2D_layer})
    χ = layers.chi
    T = (layers.n_layers == 0) ? layers.T0 : layers.layers[end].T # Get the last tensor 
    trunc_along = (layers.n_layers % 2 == 0) ? :x : :y
    T_new, U = HOTRG_2D_trunc(T,χ,trunc_along)
    T_norm = norm(T_new)
    return 2, HOTRG2D_layer(T_new/T_norm,T_norm,trunc_along,U)
end

function get_trace(layers::RG_layers{HOTRG2D_layer},n::Int)
    T = (n == 0) ? layers.T0 : layers.layers[n].T 
    @tensor TrT = T[i,j,i,j]
    return TrT
 end

function HOTRG_2D_trunc(T::TensorMap,χ::Int,along::Symbol)
    if along == :x 
        return HOTRG_2D_trunc_x(T,χ)
    elseif along == :y 
        return HOTRG_2D_trunc_y(T,χ)
    else
        throw("$along is not a valid direction")
    end
end

function HOTRG_2D_trunc_x(T::TensorMap,χ::Int) # Contract along y 
    @tensor T[la,lb,k,ja,jb,i] := T[la,m,ja,i]*T[lb,k,jb,m]
    @tensor TR[jc,jd,ja,jb] := conj(T[la,lb,k,jc,jd,i])*T[la,lb,k,ja,jb,i]
    @tensor TL[la,lb,lc,ld] := T[la,lb,k,ja,jb,i]*conj(T[lc,ld,k,ja,jb,i])

    UR,_,_,ϵR = tsvd(TR,((1,2),(3,4)),trunc=truncdim(χ));
    UL,_,_,ϵL = tsvd(TL,((1,2),(3,4)),trunc=truncdim(χ));
    
    U = (ϵR > ϵL) ? UL : UR
    @tensor T_trunc[l,k,j,i] := conj(U[la,lb,l])*T[la,lb,k,ja,jb,i]*U[ja,jb,j]
    return T_trunc,U # U has the shape ← (=▷-) ←
end

function HOTRG_2D_trunc_y(T::TensorMap,χ::Int) # Contract along x
    @tensor T[l,ka,kb,j,ia,ib] := T[l,ka,m,ia]*T[m,kb,j,ib]
    @tensor TT[ic,id,ia,ib] := conj(T[l,ka,kb,j,ic,id])*T[l,ka,kb,j,ia,ib]
    @tensor TB[kc,kd,ka,kb] := T[l,kc,kd,j,ia,ib]*conj(T[l,ka,kb,j,ia,ib])

    UT,_,_,ϵT = tsvd(TT,((1,2),(3,4)),trunc=truncdim(χ));
    UB,_,_,ϵB = tsvd(TB,((1,2),(3,4)),trunc=truncdim(χ));

    U = (ϵB > ϵT) ? UT : UB
    @tensor T_trunc[l,k,j,i] := conj(U[ka,kb,k])*T[l,ka,kb,j,ia,ib]*U[ia,ib,i]
    return  T_trunc,U # U has the shape ← (=▷-) ←
end 
