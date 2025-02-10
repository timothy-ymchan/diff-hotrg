using TensorKit


function SPT_Chain(;U,B,symmetry=:none)
    # SPT chain in Eq.29 https://arxiv.org/pdf/0903.1069
    if symmetry == :none 
        V = ℂ^3 # Spin 1 system
        # Spin matrices (Somehow MPSKItModels don't have identity operator)
        Sx = TensorMap(zeros,ComplexF64,V←V)
        Sy = TensorMap(zeros,ComplexF64,V←V)
        Sz = TensorMap(zeros,ComplexF64,V←V)
        eye = id(V)
        
        Sx.data .= ([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] ./ sqrt(2))
        Sy.data .= ([0.0 -1.0*im 0.0; 1.0*im 0 -1.0*im; 0.0 1.0*im 0.0]) ./ sqrt(2)
        Sz.data .= [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]


        return Sx⊗Sx+Sy⊗Sy+Sz⊗Sz + 0.5*U*((Sx*Sx)⊗eye + eye⊗(Sx*Sx)) + 0.5*B*(eye⊗Sz+Sz⊗eye)
    elseif symmetry == :Z2 
        V = Z2Space(0=>1,1=>2)
        H0 = SPT_Chain(;U=U,B=B,symmetry=:none)
        return project_hamiltonian(H0,V⊗V←V⊗V,SPT_Chain_Z2_Projector)
    else
        throw("$symmetry is not implemented")
    end

end

# Projection operator for Hamiltonians
function project_hamiltonian(ℋ0,space,projector)
    ℋsym = TensorMap(zeros,ComplexF64,space)
    # Projection onto symmetric subspace 
    for (f_image,f_domain) in fusiontrees(ℋsym)
        P_image = otimes(projector.(f_image.uncoupled)...) # ⊗_i P_i for i ∈ f_image's uncoupled sectors
        P_domain = otimes(projector.(f_domain.uncoupled)...) # ⊗_i P_i for i ∈ f_domain's uncoupled sectors
        
        ℋ_block = block(P_image*ℋ0*adjoint(P_domain),Trivial()) # Project ℋ0 to charged blocks
        
        # ℋ[f_image,f_domain] has shape (dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled)...)
        ℋsym[f_image,f_domain] .= reshape(ℋ_block,size(ℋsym[f_image,f_domain])) 
    end
    return ℋsym
end


function SPT_Chain_Z2_Projector(charge::Z2Irrep)
    # Basis convention:
    # i=1 → Jz = 1; i=2 → Jz=0; i=3 → Jz=-1
    if charge == Irrep[ℤ₂](0)
        projector = TensorMap(zeros,ℂ^1←ℂ^3)
        projector[2] = 1 # Jz = 0
    elseif charge == Irrep[ℤ₂](1)
        projector = TensorMap(zeros,ℂ^2←ℂ^3) 
        projector[1,1] = 1 # Jz = 1 
        projector[2,3] = 1 # Jz = -1
    else 
        throw("$charge is not a valid ℤ₂ charge!")
    end
    return projector
end

function A4_Chain(;λ,μ)
    V = ℂ^3 # Spin 1 system
    # Spin matrices (Somehow MPSKItModels don't have identity operator)
    Sx = TensorMap(zeros,ComplexF64,V←V)
    Sy = TensorMap(zeros,ComplexF64,V←V)
    Sz = TensorMap(zeros,ComplexF64,V←V)
    eye = id(V)
    
    Sx.data .= ([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] ./ sqrt(2))
    Sy.data .= ([0.0 -1.0*im 0.0; 1.0*im 0 -1.0*im; 0.0 1.0*im 0.0]) ./ sqrt(2)
    Sz.data .= [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
    H0 = Sx⊗Sx + Sy⊗Sy + Sz⊗Sz # The Heisenberg term
    H1 = (Sx⊗Sx)^2 + (Sy⊗Sy)^2 + (Sz⊗Sz)^2 # The square terms
    H2 = (Sx*Sy)⊗Sz + (Sz*Sx)⊗Sy + (Sy*Sz)⊗Sx +
        (Sy*Sx)⊗Sz + (Sx*Sz)⊗Sy + (Sz*Sy)⊗Sx + 
        Sx⊗(Sy*Sz) + Sz⊗(Sx*Sy) + Sy⊗(Sz*Sx) + 
        Sx⊗(Sz*Sy) + Sz⊗(Sy*Sx) + Sy⊗(Sx*Sz) # The complicated term

    term = H0 + μ*H1 + λ*H2 # The A4 Hamiltonian 
    return term 
end

function HeisenbergXXZ_Chain(;J,g)
    throw("Not yet implemented")
end