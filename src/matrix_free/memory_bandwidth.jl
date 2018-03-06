using BenchmarkTools

make_A(n, k) = sprand(n, n, (2k + 1) / n)
make_B(n, k) = spdiagm([rand(n - abs(i)) for i = -k:k], -k:k)

function memory_test(n = 3_000_000, k = 4)
    fst = @benchmark A_mul_B!(y, A, x) setup = (A = make_A($n, $k); x = ones($n); y = similar(x))
    snd = @benchmark A_mul_B!(y, A, x) setup = (A = make_B($n, $k); x = ones($n); y = similar(x))

    fst, snd
end