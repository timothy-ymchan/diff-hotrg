using TensorKit
using LinearAlgebra

include("trotter_sukuzi.jl")

# Two site iTEBD algorithm 
function Two_Site_Op_TEBD(MPO::Vector{TT},χ_max::Int) where TT <: TensorMap
    # Perform time evolution by zipping up the MPOs, doing local truncation if necessary
    # Idea: 
    # Tb Ta      --[ ]__[ ]--
    # Tb Ta      --[ ]  [ ]--
    # Then swap and do SVD again 
     
    Ta, Tb = MPO[1], MPO[2]
    @tensor Ω[ia,ib,ja,jb,ka,kb,la,lb] := Tb[ia,m,n,la]*Ta[ib,ja,o,m]*Tb[n,p,ka,lb]*Ta[o,jb,kb,p]

    Ωa,Ωb, _ = SVD_factorize(Ω,(1,7,8,5),(2,3,4,6),truncdim(χ_max)) # (ia,la,lb,ka) (ib,ja,jb,kb) => (ia,la,lb,ka,m) (m,ib,ja,jb,kb)
    @tensor Ω[ia,ib,m,ka,kb,n] := Ωb[n,ia,aa,bb,ka]*Ωa[ib,aa,bb,kb,m]
    Ta,Tb, _ = SVD_factorize(Ω,(1,6,4),(2,3,5),truncdim(χ_max)) # (ia,l,ka) (ib,j,kb) =>  (ia,l,ka,jj) (ll,ib,j,kb) 
    @tensor Ta[i,j,k,l] := Ta[i,l,k,j]
    @tensor Tb[i,j,k,l] := Tb[l,i,j,k]

    return Ta,Tb
end    

function Gu_ITEBD_Step(U,χ)
    Ta,Tb = SVD_factorize(U,(1,3),(2,4),notrunc())
    @tensor Ω1[i,j,k,l] := Ta[i,m,j]*Tb[l,m,k]
    @tensor Ω2[i,j,k,l] := Tb[l,i,m]*Ta[m,k,j]
    La,Lb = SVD_factorize(Ω1,(1,4),(2,3),truncdim(χ)) # (i,l,σ), (σ,j,k)
    Ra,Rb = SVD_factorize(Ω2,(1,2),(3,4),truncdim(χ)) # (i,j,σ), (σ, k,l)
    #@tensor begin
    @tensor LbRb[i,j,m,n] := Lb[i,jj,n]*Rb[j,m,jj]
    @tensor LbRbTaTb[i,j,o,p] := LbRb[i,j,m,n]*U[n,m,o,p]
    @tensor RaLa[o,p,k,l] := Ra[o,jj,l]*La[p,jj,k]
    @tensor LbRbTaTbRaLa[i,j,l,k] := LbRbTaTb[i,j,o,p]*RaLa[o,p,k,l]
    #end
    return LbRbTaTbRaLa
end

function Σ_Spectrum(T::TensorMap,legs1,legs2,sector=Trivial(),num_Σ=5)
    _,Σ,_,_ = tsvd(T,legs1,legs2)
    mat = block(Σ,Trivial())
    return diag(mat)[1:num_Σ]
end
