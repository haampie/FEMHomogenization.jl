using FEMHomogenization

function run()
    FEMHomogenization.coarse_grid_with_local_refinement()
end

println(run())