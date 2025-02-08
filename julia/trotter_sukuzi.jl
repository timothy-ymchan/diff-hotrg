using TensorKit

function SVD_factorize(op::TensorMap,ind_1,ind_2,trunc::TruncationScheme)
    U,Σ,V,ϵ = tsvd(op,ind_1,ind_2,trunc=trunc)
    Λ = sqrt(Σ)
    Sa = U*Λ
    Sb = Λ*V
    return Sa,Sb,ϵ # Sa[i,m] Sb[m,j] = op[i,j]
end

function Two_Site_Trotter_Suzuki(H_odd::TensorMap,H_even::TensorMap,Δt::Complex)
    # First order Trotter Suzuki (See https://arxiv.org/pdf/1801.00719 for the algorithm)
    U1 = exp(-Δt*im*H_odd)   # U[oa,ob,ia,ib] where actually it is V⊗V ← V⊗V because of conventions of matrix multiplication
    U2 = exp(-Δt*im*H_even) 
    S1a, S1b, _ = SVD_factorize(U1,(1,3),(2,4),notrunc()) # (oa,ia,m) - (m,ob,ib)
    S2a, S2b, _ = SVD_factorize(U2,(1,3),(2,4),notrunc())
    
    @tensor Ta[i,j,k,l] := S2b[l,i,m]*S1a[m,k,j]
    @tensor Tb[i,j,k,l] := S2a[i,m,j]*S1b[l,m,k]
    return Ta,Tb
end
