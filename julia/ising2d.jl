include("models.jl")
using TensorKit 
using QuadGK

struct Ising2D <: Model 
    N0::Int
    β::Real
    J::Real 
    H::Real 
    Ising2D(;β,J,H) = new(1,β,J,H)
end

function T0(model::Ising2D,symmetry=:none)

    #  δ[1,2,3,4]
    #                             ↓
    #                            (W)
    #                             ↓ 4
    # T[1,2,3,4] =  ← (W†) ← 1 ← (δ) ← 3 ← (W) ←
    #                             ↓ 2
    #                            (W†)
    #                             ↓

    # Hamiltonian: ℍ = ∑ -J σᵢσⱼ + H σᵢ

    βJ,βH = model.β*model.J, model.β*model.H
    cc,ss = sqrt(cosh(βJ)),sqrt(sinh(βJ))
    if symmetry == :none
        W = TensorMap(zeros,ComplexF64,ℂ^2,ℂ^2)
        δ = TensorMap(zeros,ComplexF64,ℂ^2⊗ℂ^2,ℂ^2⊗ℂ^2)

        # δ tensor
        δ[1,1,1,1] = exp(-βH) # ↑
        δ[2,2,2,2] = exp(+βH) # ↓

        # W tensor ←(W)←
        W[1,1] = cc
        W[2,1] = cc 
        W[1,2] = ss 
        W[2,2] = -ss
        
        @tensor T[i,j,k,l] := δ[ii,jj,kk,ll]*W[kk,k]*W[ll,l]*conj(W[ii,i])*conj(W[jj,j])
        
        return T
    elseif symmetry == :Z2
        model.H == 0 || throw("H≠0 does not have Z2 symmetry")
        V = Z2Space(0=>1,1=>1)
        K = TensorMap(zeros,ComplexF64,V,V)
        δ = TensorMap(ones,ComplexF64,V⊗V,V⊗V)/2
        # We are using a gauge transformed tensor by inserting H gates around W (i.e. K=HWH)
        # In this case the W is manifestly symmetric and we have K |+> = √2cosh(x) |+> and K|-> = √2sinh(x) |->
        block(K,Z2Irrep(0))[1] = sqrt(2)*cc 
        block(K,Z2Irrep(1))[1] = sqrt(2)*ss 
        @tensor T[i,j,k,l] := δ[ii,jj,kk,ll]*K[kk,k]*K[ll,l]*conj(K[ii,i])*conj(K[jj,j])
        return T
    else
        throw("$symmetry is not implemented")
    end
end

function lnz(model::Ising2D,atol=1e-10)
    model.H == 0 || throw("Analytical solution only exist for H=0")
    K = model.β*model.J
    k = sinh(2*K)^(-2)
    I,ϵ = quadgk(θ->log(cosh(2*K)*cosh(2*K)+sqrt(1+k^2-2*k*cos(θ))/k),0,π,atol=atol)
    ϵ < 1e-5 || throw("Integration error: ϵ is too large $ϵ")
    return log(2)/2 + I/(2*π)
end

function Tc(model::Ising2D)
    model.H == 0 || throw("Analytical solution only exist for H=0")
    return 2*model.J/log(1+sqrt(2))
end