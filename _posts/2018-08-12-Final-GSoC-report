---
title:  "GSoC - Final Report"
date:   2018-08-12
categories: text
---

In this blog post, I will talk about what I did in the 3 months of the Google Summer of Code.

## **Locally optimal preconditioned conjugate gradient (LOBPCG)**

During this GSoC project I completed a number of sub-projects. Firstly, I implemented the locally optimal block preconditioned conjugate gradient method (LOBPCG) as described in [this paper](https://epubs.siam.org/doi/abs/10.1137/S1064827500366124?journalCode=sjoce3&) which finds extremal generalized eigenvalues and their eigenvectors of the system `Ax = λ Bx`, where `A` and `B` are Hermitian matrices and `B` is positive definite. A few passes of code writing, debugging and cleaning were needed to port the [standard implementation](https://github.com/scipy/scipy/blob/v1.1.0/scipy/sparse/linalg/eigen/lobpcg/lobpcg.py#L109-L568) of the algorithm that can be found in the Scipy module of Python. An idiomatic Julia version of the algorithm was written that is optimized to eliminate unnecessary memory allocations making heavy use of the Julia macro `@views`. This algorithm was submitted as a PR to [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl) and later merged.

## **Buckling**

After programming the LOBPCG algorithm to find generalized eigenvalues and testing it on dummy random matrices, it was time to test it on a real life problem. So I developed a linear elasticity analysis package called [LinearElasticity.jl](https://github.com/mohamed82008/LinearElasticity.jl) which can do stress and buckling analysis of linear elastic structures. Heavy use of [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) was made and [Makie.jl](https://github.com/JuliaPlots/Makie.jl) was used for visualization of the mode shapes. Results were then shared in the blog post [The Buckle Dance](https://mohamed82008.github.io/ScienceLounge/text/2018/07/11/The-buckle-dance/).

## **Preconditioners**

The third component of the package was to develop and combine a set of common preconditioners into a package which I called [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl). The package combines diagonal preconditioner, incomplete Cholesky preconditioner from the package [IncompleteSelectedInversion.jl](https://github.com/ettersi/IncompleteSelectedInversion.jl) and the algebraic multigrid (AMG) preconditioner from the package [AlgebraicMulrigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl), formerly known as `AMG.jl`. In order to validate the AMG package, I read the majority of its lines of code and found a few bugs for unsymmetric matrices and unhandled corner cases which I fixed in [this PR](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/pull/43).

I then profiled the memory and time requirements of different parts of the program to identify bottlenecks. Originally, the plan was to parallelize the splitting operator which determines coarse and fine variables, however this part was found to take only 2% of the operator building time and the majority of the time was spent in the so-called Galerkin projection that takes the form `A' B A` where `A` and `B` are sparse matrices. So I shifted my focus to making the AMG solver and preconditioner more lean by eliminating all allocations in the functions used during the solve process and parallelizing the smoother.

I also conducted experiments using the AMG preconditioner to perform stress and buckling analysis using LinearElasticity.jl and results were not satisfactory. While there was a significant reduction in the number of iterations of these algorithms, this reduction was not enough to compensate the additional work done in the AMG operator's "smoother", which was called twice in every preconditioner call. This led the preconditioned conjugate gradient (CG) to do around 25% more matrix-vector multiplications worth of work in total than the normal CG. For the LOBPCG, it was slightly better simply because each LOBPCG iteration does 2 matrix-vector multiplications and is therefore more expensive. But still the preconditioned and unpreconditioned versions were doing a comparable number of matrix-vector multiplications.

The AMG PR is still pending as I am trying to fix some tests and support both the stable v0.6.4 version of the Julia language and the more recent v1.0.0. Eliminating allocations led to some errors which I am working on fixing. And eliminating dependencies which do not yet work with Julia v1.0.0 led to some other precision errors as I was writing the test sparse matrices in .jl files as opposed to the .jld2 files which are not yet readable in Julia v1.0.0. A couple more days at most are needed to complete the AMG PR which will hopefully be merged soon after.
