function basis_coefficients(p::Expr, x::NTuple{nbasis,NTuple{dim,T}}, vars::NTuple{dim,Symbol}) where {nbasis,dim,T}
    A = zeros(nbasis, nbasis)
    sandbox = Module()

    for i = 1:nbasis, j = 1:nbasis

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
    simplify(:(0.0 * f(3x) + 1.0 * 3.0)) -> 3.0

Simplify an expression
"""
function simplify(e::Expr)
    _simplify(e)
end

_simplify(e) = e

function _simplify(e::Expr)
    if e.head == :call
        op = e.args[1]
        args = [ _simplify(arg) for arg in e.args[2:end] ]
        
        if op == :*
            0 in args && return 0
            filter!(x -> x != 1, args)
            length(args) == 1 && return args[1]
            length(args) == 0 && return 1
        elseif op == :+
            filter!(x -> x != 0, args)
            length(args) == 1 && return args[1]
            length(args) == 0 && return 0
        end

        return Expr(:call, op, args...)
    end

    return e
end

function polynomials(p::Expr, A::AbstractMatrix)
    Ainv = inv(A)
    nbasis = size(A, 1)
    basis_functions = []

    for j = 1 : nbasis
        args = [:($(Ainv[i,j]) * $(p.args[i + 1])) for i = 1 : nbasis]
        push!(basis_functions, simplify(Expr(:call, :+, args...)))
    end

    return basis_functions
end

function create_basis(p::Expr, coords::NTuple{nbasis,NTuple{dim,T}}, vars::NTuple{dim,Symbol}) where {nbasis,dim,T}
    return polynomials(p, basis_coefficients(p, coords, vars))
end

create_basis_gradients(ϕs, vars) = [[differentiate(ϕ, x) for x in vars] for ϕ in ϕs]
