using Base.Test
using FEM

function test_basis_functions_on_nodes(ϕs, nodes, vars)
    sandbox = Module()

    # Test whether ϕᵢ(nⱼ) = δᵢⱼ
    for i = 1 : 3, j = 1 : 3
        for (var, val) in zip(vars, nodes[j])
            eval(sandbox, :($var = $val))
        end
        value = eval(sandbox, ϕs[i])
        @test i == j ? value == 1 : value == 0
    end
end

@testset "Creating basis functions" begin
    @testset "2D example" begin
        nodes = ((0., 0.), (1., 0.), (0., 1.))
        vars = (:u, :v)
        ϕs = FEM.create_basis(:(1 + u + v), nodes, vars)
        test_basis_functions_on_nodes(ϕs, nodes, vars)
    end

    @testset "3D example" begin
        nodes = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        vars = (:u, :v, :w)
        ϕs = FEM.create_basis(:(1 + u + v + w), nodes, vars)
        test_basis_functions_on_nodes(ϕs, nodes, vars)
    end

    @testset "2D hp-FEM" begin
        nodes = ((0., 0.), (1., 0.), (0., 1.), (0.5, 0.5))
        vars = (:u, :v)
        ϕs = FEM.create_basis(:(1 + u + v + u*v), nodes, vars)
        test_basis_functions_on_nodes(ϕs, nodes, vars)
    end
end