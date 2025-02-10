using TensorKit

function LoopTNR_renormalize(T::TensorMap,χ::Int)
    # The code can be simplified if we have rotation symmetry constraints, but we will implement the general one
    # We will assume, however, that T1=T2=T3=T4, as tensors 

    Ta,Tb = entanglement_filtering(T,T)
    T1,T2,T3,T4 = get_loop_tensors(Ta,Tb)
    Tr1,Tr2,Tr3,Tr4,Tr5,Tr6,Tr7,Tr8 = get_trg_guess(T1,T2,T3,T4,χ)
end


function loop_mps_approx(TT_guess::Vector{TT},TT_target::Vector{TT},η=1e-10) where TT <: TensorMap
    # Remark: This is actually an ill-defined problem because of gauge fixing issues 
    
end 

function get_Wi(TT_target::Vector{TT},TT_guess::Vector{TT},pos::Int) where TT <: TensorMap 
    # Calculate the Wᵢ environment with the pos-th tensor in TT_guess removed 
    # Do the zip-up algorithm
    length(TT_guess) == 2*length(TT_target) || throw("We require length(TT_guess)=2*length(TT_target)")
    # The returned leg signature Wi[i,j,k]
    # -- (G1) -- (G2) -- ... -- (Gi-1) -- i  k -- (G2n) --
    #     \       /                        / j
    # -------(T1)------- (T2) -------

    if pos % 2 == 1
        # Reverse the order of the tensors, zip up, and then permute
    end
    # Now permute the tensors such that the TT_guess[end] is always the tensor that we leave out 
    # We zip from right to left
    T = # Some original tensor
    for i in range(1,legnth(TT_guess)//2-1)
        Gᵢ = TT_guess[2*i-1]
        Gᵢ₊₁ = TT_guess[2*i-1]
        Tᵢ = TT_target[i]
        @tensor T[i,j,k,l] := T[i,j,kk,ll]*Gᵢ[kk,nn,mm]*Gᵢ₊₁[mm,oo,k]*conj(Tᵢ[ll,nn,oo,l])
    end 
    
    # Contract the last tensors and loop up the loosing ends at the bottom

end


function get_Ni(TT_target::Vector{TT},TT_guess::Vector{TT},pos::Int) where TT <: TensorMap 
    # Calculate the Nᵢ environment with the pos-th tensor in TT_guess removed
end 

function get_loop_tensors(Ta::TensorMap,Tb::TensorMap)
    # @tensor T1[i,j,k,l] := Ta[i,l,j,k]
    # @tensor T2[i,j,k,l] := Tb[l,k,j,i]
    # @tensor T3[i,j,k,l] := Ta[k,j,i,l]
    # @tensor T4[i,j,k,l] := Tb[j,i,l,k]
    @tensor T1[i,j,k,l] := Ta[k,i,j,l]
    @tensor T2[i,j,k,l] := Tb[i,j,l,k]
    @tensor T3[i,j,k,l] := Ta[j,l,k,i]
    @tensor T4[i,j,k,l] := Tb[l,k,i,j]
    return T1,T2,T3,T4
end


function Levin_TRG(T::TensorMap,ind1::Tuple{Int},ind2::Tuple{Int},χ::Int)
    U,Σ,V,_ = tsvd(T,ind1,ind2,trunc=truncdim(χ))
    Λ = sqrt(Σ)
    @tensor Ta[i,j,k] := U[i,j,kk]*Λ[kk,k]
    @tensor Tb[i,j,k] := Λ[i,ii]*V[ii,j,k]
    return Ta,Tb
end

function get_trg_guess(T1::TensorMap,T2::TensorMap,T3::TensorMap,T4::TensorMap,χ::Int)
    # Get TRG tensors via SVD and return the 8 tensors 
    #     |      |
    # - (T4) - (T1) -
    #     |      |
    # - (T3) - (T2) -
    #     |      |
    T1a,T1b = Levin_TRG(T1,(1,4),(3,2),χ)
    T2a,T2b = Levin_TRG(T2,(4,3),(2,1),χ)
    T3a,T3b = Levin_TRG(T3,(3,2),(1,4),χ)
    T4a,T4b = Levin_TRG(T4,(2,1),(4,3),χ)

    return T1a,T1b,T2a,T2b,T3a,T3b,T4a,T4b
end

function entanglement_filtering(Ta::TensorMap,Tb::TensorMap)
    # The MPS type tensors has the following leg labels
    #      j\  /k
    # i -- ( Ti ) -- l
    
    T1,T2,T3,T4 = get_loop_tensors(Ta,Tb)

    # Entanglement filtering 
    L1,R4 = get_L([T1,T2,T3,T4]), get_R([T1,T2,T3,T4])
    P1L,P4R = get_projectors(L1,R4)

    L2,R1 = get_L([T2,T3,T4,T1]), get_R([T2,T3,T4,T1])
    P2L,P1R = get_projectors(L2,R1)

    L3,R2 = get_L([T3,T4,T1,T2]), get_R([T3,T4,T1,T2])
    P3L,P2R = get_projectors(L3,R2)

    L4,R3 = get_L([T4,T1,T2,T3]), get_R([T4,T1,T2,T3])
    P4L,P3R = get_projectors(L4,R3)

    @tensor Ta_new[i,j,k,l] := Ta[ii,jj,kk,ll]*P1L[i,ii]*P1R[j,jj]*P3L[kk,k]*P3R[ll,l]
    @tensor Tb_new[i,j,k,l] := Tb[ii,jj,kk,ll]*P2R[i,ii]*P4L[j,jj]*P4R[kk,k]*P2L[ll,l]

    return Ta_new, Tb_new
end

function get_projectors(L,R,η=1e-12)
    @tensor LR[i,j] := L[i,k]*R[k,j]
    U,Λ,Vʰ,ϵ = tsvd(LR,(1,),(2,),trunc=truncbelow(η))
    Σ = inv(sqrt(Λ))
    @tensor Pr[i,j] := R[i,jj]*conj(Vʰ[kk,jj])*Σ[kk,j]
    @tensor Pl[i,j] := Σ[i,jj]*conj(U[kk,jj])*L[kk,j]
    return Pl,Pr 
end

function next_L(L::TensorMap,TT::Vector{TM}) where TM <: TensorMap
    for i in range(1,length(TT))
        T = TT[i]
        @tensor LTi[i,j,k,l] := L[i,ii]*T[ii,j,k,l]
        _,L = leftorth(LTi,(1,2,3),(4,))
    end
    return L/norm(L)
end 

function get_L(TT::Vector{TM},ϵ=1e-10,max_iter=100) where TM <: TensorMap 
    # Again assume this charge 
    #      j\  /k
    # i -- ( Ti ) -- l
    W = space(TT[1],1) # Get space for leftmost index
    Lᵢ = id(W)
    iter = 0
    conv = false 
    ϵᵢ = Inf
    while iter < max_iter && !conv
        Lᵢ₊₁ = next_L(Lᵢ,TT)
        ϵᵢ = norm(Lᵢ₊₁-Lᵢ)
        conv = ϵᵢ <= ϵ
        Lᵢ = Lᵢ₊₁
        iter += 1 
    end 
    iter != max_iter || print("get_L: max_iter reached. ϵᵢ = ",ϵᵢ)
    return Lᵢ
end

function next_R(R::TensorMap,TT::Vector{TM}) where TM <: TensorMap 
    for i in range(0,length(TT)-1)
        T = TT[end-i]
        @tensor TiR[i,j,k,l] := T[i,j,k,ll]*R[ll,l]
        R,_ = rightorth(TiR,(1,),(2,3,4))
    end
    return R/norm(R)
end

function get_R(TT::Vector{TM},ϵ=1e-10,max_iter=100) where TM <: TensorMap 
    # Again assume this charge 
    #      j\  /k
    # i -- ( Ti ) -- l
    W = space(TT[end],4)
    Rᵢ = id(W')
    iter = 0
    conv = false
    ϵᵢ = Inf
    while iter < max_iter && !conv
        Rᵢ₊₁ = next_R(Rᵢ,TT)
        ϵᵢ = norm(Rᵢ₊₁-Rᵢ)
        conv = ϵᵢ <= ϵ
        Rᵢ = Rᵢ₊₁
        iter += 1
    end
    iter != max_iter || print("get_R: max_iter reached. ϵᵢ = ",ϵᵢ)
    return Rᵢ
end