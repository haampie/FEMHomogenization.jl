using BenchmarkTools

function make_A(n, k)
    A = sprand(n, n, k / n)
end

function make_B(n, k)
    diags = [rand(n - i + 1) for i = 1 : k]
    return spdiagm(diags, ((0:k-1)...,)) 
end

function memory_test(n = 5_000_000, k = 9)

    fst = @benchmark A_mul_B!(y, A, x) setup = (A = make_A($n, $k); x = ones($n); y = similar(x)) gcsample = true
    snd = @benchmark A_mul_B!(y, A, x) setup = (A = make_B($n, $k); x = ones($n); y = similar(x)) gcsample = true

    fst, snd
end