using FEMHomogenization

function run()
    FEMHomogenization.example_one()
    Profile.clear_malloc_data()
    @profile FEMHomogenization.example_one()
end

run()