using BenchmarkTools
using AMD

function merge_triangles(n = 8)
    # Assume right-angled triangles with side divided in n parts
    # so there are m = sum(1 : n + 1) unknowns
    # We number from the right angle counter-clockwise, except for the inner triangle (?)

    # Corners are 1, n + 1, 2n + 1
    # Side interiors are 2:n, n+2:2n, 2n+2:3n

    m = sum(1 : n + 1)

    top = collect(linspace(1, m, m))
    mid = collect(linspace(1, m, m))
    left = collect(linspace(1, m, m))
    right = collect(linspace(1, m, m))

    # New triangle has side length 2n - 1
    M = sum(1 : 2n - 1)
    V = Vector{Float64}(M)

    idx = 0

    ## Fix the boundary

    ## resp: south of left, east of right, morth of top, west of left
    copy!(V, 1:n, left, 1:n)
    copy!(V, n+1:3n, right, 1:2n)
    copy!(V, 3n+1:5n, top, n+1:3n)
    copy!(V, 5n+1:6n, left, 2n+1:3n)

    # Sum the edges in our middle triangle
    @simd for i = 1 : n + 1
        mid[i] += right[2n+i]
    end

    @simd for i = 1 : n + 1
        mid[n + i] += left[n + i]
    end

    @simd for i = 1 : n
        mid[2n + i] += top[i]
    end

    # Fix the last item
    mid[1] += top[n + 1]


    V
end

function recursive_test(refs = 4)
    bilinearform = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    nodes = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    elements = SVector{3,Int64}[(1, 2, 3)]
    base = Mesh(Tri, nodes, elements)
    A = assemble_matrix(base, bilinearform)
    meshes = Vector{typeof(base)}(refs)
    matrices = Vector{typeof(A)}(refs)

    meshes[1] = base
    matrices[1] = A

    for i = 2 : refs
        meshes[i] = refine(meshes[i-1], remove_duplicates!(to_graph(meshes[i-1])))
        matrices[i] = assemble_matrix(meshes[i], bilinearform)
    end

    matrices, meshes
end

function bench_things(r)
    As, Ms = recursive_test(r)

    timings_1 = []
    timings_2 = []
    nonzeros = Int[]

    for (i, A) = enumerate(As)
        println(i)
        @time p = amd(A)
        B = A[p,p]
        x = rand(size(A, 1))
        y = similar(x)
        push!(timings_1, @benchmark A_mul_B!($y, $A, $x))
        push!(timings_2, @benchmark A_mul_B!($y, $B, $x))
        push!(nonzeros, nnz(A))
    end

    for (i, (t1, t2, b)) in enumerate(zip(timings_1, timings_2, nonzeros))
        println(i, "\t", median(t1).time / b, "\t", median(t2).time / b)
    end 
    
    return timings_1, timings_2, nonzeros
end