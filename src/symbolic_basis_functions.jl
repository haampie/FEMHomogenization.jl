"""
    simplify(:(0.0 * f(3x) + 1.0 * 3.0)) -> 3.0

Simplify an expression
"""
simplify(e::Expr) = _simplify(e)

_simplify(e) = e

function _simplify(e::Expr)
    if e.head == :call
        op = e.args[1]
        args = [_simplify(arg) for arg in e.args[2:end]]
        
        if op == :*
            # Fold numbers
            prod = mapreduce(x -> isa(x, Number) ? x : 1, *, 1, args)
            filter!(x -> !isa(x, Number), args)

            # Simplify
            if prod == 0
                return 0
            end

            if length(args) == 0
                return prod
            end

            if abs(prod) == 1
                if length(args) == 1
                    if prod == 1
                        return args[1]
                    else
                        return Expr(:call, :-, args[1])
                    end
                else
                    return Expr(:call, :*, args...)
                end
            end

            return Expr(:call, :*, prod, args...)
        elseif op == :+
            sum = mapreduce(x -> isa(x, Number) ? x : 0, +, 0, args)
            filter!(x -> !isa(x, Number), args)

            if length(args) == 0
                return sum
            end

            if sum == 0
                if length(args) == 1
                    return args[1]
                else
                    return Expr(:call, :+, args...)
                end
            end

            return Expr(:call, :+, sum, args...)
        else
            return Expr(:call, op, args...)
        end
    end

    return e
end

function basis_coefficients(p::Expr, x::NTuple{n,NTuple{d,T}}, vars::NTuple{d,Symbol}) where {n,d,T}
    A = zeros(n, n)

    sandbox = Module()

    for i = 1:n, j = 1:n

        # Fill in the variables
        for (variable, value) = zip(vars, x[j])
            eval(sandbox, :($variable = $value))
        end

        # Evaluate the expression
        A[j,i] = eval(sandbox, p.args[i + 1])
    end

    return A
end

"""
Construct the basis functions in the nodes
"""
function polynomials(p::Expr, A::AbstractMatrix)
    Ainv = inv(A)
    n = size(A, 1)
    basis_functions = []

    for j = 1 : n
        args = [:($(Ainv[i,j]) * $(p.args[i + 1])) for i = 1 : n]
        push!(basis_functions, simplify(Expr(:call, :+, args...)))
    end

    return basis_functions
end


function create_basis(p::Expr, nodes::NTuple{n,NTuple{d,T}}, vars::NTuple{d,Symbol}) where {n,d,T}
    return polynomials(p, basis_coefficients(p, nodes, vars))
end

create_basis_gradients(ϕs, vars) = [[differentiate(ϕ, x) for x in vars] for ϕ in ϕs]

"""
Evaluate the basis in the given node
"""
function evalute_basis(ϕ::Expr, ∇ϕ::Vector{Expr}, vars::NTuple{d,Symbol}) where {d}
    unpacking = Expr(:block)

    for (idx, var) in enumerate(vars)
        push!(unpacking.args, :($var = node[$idx]))
    end

    ∇ϕ_eval = :(SVector())

    for ∂ϕ in ∇ϕ
        push!(∇ϕ_eval.args, ∂ϕ)
    end

    return quote
        function testtest(node::NTuple{n,T}) where {n,T}
            $unpacking
            $ϕ, $∇ϕ_eval
        end
    end
end

struct ReferenceBasis{d}
    ϕ::Float64
    ∇ϕ::SVector{d,Float64}
end