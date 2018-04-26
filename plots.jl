using Plots

function plot_stuff()
    with_n4 = [readdlm("h_refinement_refs_$(d)_n_4_times_100") for d = 2 : 4]
    with_n5 = [readdlm("h_ref_n_5_ref_$(d).txt") for d = 1 : 3]

    p = plot(title = "Domain 16x16 different FEM refinements -- 100 runs", ylabel = "sqrt(mean(err^2))", xlabel = "steps", legend = :top, ylims = (0.05, 1.0), yscale = :log10)
    plot!([sqrt(mean((with_n4[1][:, i] .- 2.0).^2)) for i = 1 : 4], label = "64x64 FEM", mark = :auto)
    plot!([sqrt(mean((with_n4[2][:, i] .- 2.0).^2)) for i = 1 : 4], label = "128x128 FEM", mark = :auto)
    plot!([sqrt(mean((with_n4[3][:, i] .- 2.0).^2)) for i = 1 : 4], label = "256x256 FEM", mark = :auto)
    plot!(2.0 .^ (0:-1:-3), label = "1 / 2^n")

    q = plot(title = "Domain 32x32 different FEM refinements -- 100 runs", ylabel = "sqrt(mean(err^2))", xlabel = "steps", legend = :top, ylims = (0.05, 1.0), yscale = :log10)
    plot!([sqrt(mean((with_n5[1][:, i] .- 2.0).^2)) for i = 1 : 5], label = "64x64 FEM", mark = :auto)
    plot!([sqrt(mean((with_n5[2][:, i] .- 2.0).^2)) for i = 1 : 5], label = "128x128 FEM", mark = :auto)
    plot!([sqrt(mean((with_n5[3][:, i] .- 2.0).^2)) for i = 1 : 5], label = "256x256 FEM", mark = :auto)
    # plot!([sqrt(mean((with_n5[4][:, i] .- 2.0).^2)) for i = 1 : 5], label = "512x512 FEM", mark = :auto)
    plot!(2.0 .^ (0:-1:-4), label = "1 / 2^n")
    
    p, q
end
