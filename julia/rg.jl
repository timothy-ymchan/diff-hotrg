# ==========================================
#             RG interface
# ==========================================

using TensorKit 
abstract type RG_layer end 

mutable struct RG_layers{L <: RG_layer}
    n_layers::Int
    chi::Int
    T0::TensorMap 
    sizes::Vector{Int}
    layers::Vector{L}

    RG_layers{L}(T0::TensorMap, chi::Int, s0::Int) where {L <:RG_layer} = new(0,chi,T0,[s0],Vector{L}())
end


function renormalize(layers::RG_layers,n_iter::Int)
    n_iter >= 0 || throw("$n_iter should be non-negative")
    for n in range(1,n_iter)
        scale, new_layer = next_layer(layers)
        layers.n_layers += 1
        push!(layers.sizes,layers.sizes[end]*scale)
        push!(layers.layers,new_layer)
    end
    return layers
end

function lnz(layers::RG_layers,n::Int)
    n <= layers.n_layers || throw("$n needs to be less than n_layers")
    n >= 0 || throw("$n needs to be positive")
    n == 0 && return get_trace(layers,0)/layers.sizes[1]
    # (lnZₙ (true) - lnZₙ)/N₀ s^n = (lnZₙ₋₁ (true) - lnZₙ₋₁)/N₀ s^n-1 + ln|Tₙ|/N₀ s^n
    diff = 0 
    for i in range(1,n)
        diff += log(layers.layers[i].norm)/layers.sizes[i+1]
    end
    return diff + get_trace(layers,n)/layers.sizes[n+1]
end


function renormalize_operator(layers::RG_layers,n::Int)
    n <= layers.n_layers || throw("$n needs to be less than n_layers")
end