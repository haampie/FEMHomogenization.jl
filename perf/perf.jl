using FEM
using BenchmarkTools

@benchmark FEM.example_assembly(10)
