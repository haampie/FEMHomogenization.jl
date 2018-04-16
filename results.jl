# With ref_coarse = 5 (i.e. 32 coarse cells)
> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 5, ref_fine = 5)
> results
5×5 Array{Float64,2}:
 2.00081  2.03912  2.04846  2.05339  2.05598
 1.93125  1.97129  1.97871  1.98038  1.98068
 1.97742  2.01562  2.02791  2.03662  2.04474
 1.93357  1.97426  1.98621  1.99059  1.99174
 1.95176  1.99693  2.01124  2.01511  2.01543
> means
5-element Array{Float64,1}:
 0.0489426
 0.0255752
 0.0279164
 0.0312834
 0.0341012

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 5, ref_fine = 4)
> results
5×5 Array{Float64,2}:
 1.97427  2.02118  2.03036  2.03322  2.03377
 1.94324  1.98757  1.99544  1.99875  1.99986
 1.92943  1.97296  1.97937  1.98159  1.98321
 1.98578  2.02749  2.04026  2.04462  2.04665
 1.92517  1.9637   1.97116  1.97438  1.97624
> means
5-element Array{Float64,1}:
 0.0541586
 0.0261079
 0.0276452
 0.0286051
 0.0288578

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 5, ref_fine = 3)
> results
5×5 Array{Float64,2}:
 1.92438  1.9614   1.96908  1.97208  1.97372
 1.89007  1.93585  1.94935  1.95436  1.95724
 1.95741  1.99652  2.00172  2.00267  2.003
 1.9322   1.97377  1.98595  1.99165  1.99456
 1.89027  1.93267  1.94043  1.94327  1.94474
> means
5-element Array{Float64,1}:
 0.0851516
 0.0465591
 0.0381316
 0.0350944
 0.0335015

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 5, ref_fine = 2)
> results
5×5 Array{Float64,2}:
 1.85257  1.88852  1.89584  1.89905  1.90071
 1.8478   1.89044  1.89818  1.89989  1.90038
 1.84873  1.89157  1.89936  1.90125  1.9018
 1.80253  1.8408   1.84802  1.85162  1.8545
 1.81608  1.85991  1.86784  1.87068  1.8715
> means
5-element Array{Float64,1}:
 0.167693
 0.127402
 0.119924
 0.11722
 0.115854

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 5, ref_fine = 1)
> results
5×5 Array{Float64,2}:
 1.61941  1.66057  1.67036  1.6745   1.67527
 1.63559  1.67443  1.68222  1.6851   1.6857
 1.61702  1.65315  1.66076  1.66236  1.66248
 1.63846  1.67998  1.68928  1.69107  1.69154
 1.62172  1.67642  1.68951  1.69447  1.6965
> means
5-element Array{Float64,1}:
 0.373662
 0.331248
 0.32177
 0.318717
 0.317936

# With ref_coarse = 4 (i.e. 16 coarse cells)
> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 4, ref_fine = 4)
> results
5×4 Array{Float64,2}:
 1.98266  2.02632  2.03632  2.03745
 2.06667  2.14959  2.1781   2.184
 2.04915  2.10076  2.10862  2.1122
 1.92328  1.96343  1.97139  1.9733
 1.98153  2.01609  2.02625  2.03255
> means
4-element Array{Float64,1}:
 0.0517455
 0.0834489
 0.0962742
 0.0996186

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 4, ref_fine = 3)
> results
5×4 Array{Float64,2}:
 1.95924  2.00643  2.03225  2.04338
 2.0034   2.05761  2.06941  2.07379
 1.98253  2.02841  2.03559  2.03719
 1.88347  1.92347  1.93001  1.93575
 1.9146   1.96613  1.97495  1.97616
> means
4-element Array{Float64,1}:
 0.0676025
 0.0472678
 0.0502995
 0.0517794

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 4, ref_fine = 2)
> results
5×4 Array{Float64,2}:
 1.87745  1.9349   1.94313  1.94496
 1.90875  1.96133  1.9721   1.97446
 1.85893  1.88788  1.89086  1.89149
 1.94127  1.97588  1.98222  1.98604
 1.88399  1.91889  1.92242  1.9254
> means
4-element Array{Float64,1}:
 0.109684
 0.0713635
 0.0667236
 0.0651421

> results, means = FEMHomogenization.theorem_one_point_two(times = 5, ref_coarse = 4, ref_fine = 1)
> results
5×4 Array{Float64,2}:
 1.62127  1.65944  1.66552  1.66647
 1.58026  1.61497  1.62236  1.62386
 1.66529  1.69617  1.70412  1.70849
 1.57572  1.61992  1.62752  1.62925
 1.62562  1.65555  1.66609  1.67223

> means
4-element Array{Float64,1}:
 0.38777
 0.352035
 0.344173
 0.341346
