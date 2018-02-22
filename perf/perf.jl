using FEMHomogenization: example_assembly
using BenchmarkTools

@benchmark example_assembly(10)
