---
title:  "GSoC - Final Report"
date:   2018-08-12
categories: text
---

In this blog post, I will talk about what I did in the 3 months of the Google Summer of Code and overall impression of the project.

# **Locally optimal preconditioned conjugate gradient (LOBPCG)**

During this GSoC project I completed a number of tasks. Firstly, I implemented the locally optimal block preconditioned conjugate gradient method (LOBPCG) as described in [this paper](https://epubs.siam.org/doi/abs/10.1137/S1064827500366124?journalCode=sjoce3&) which finds extremal generalized eigenvalues and their eigenvectors of the system `Ax = λ Bx`, where `A` and `B` are Hermitian matrices and `B` is positive definite. A few passes of code writing, debugging and cleaning were needed to port the [standard implementation](https://github.com/scipy/scipy/blob/v1.1.0/scipy/sparse/linalg/eigen/lobpcg/lobpcg.py#L109-L568) of the algorithm that can be found in the Scipy module of Python. An idiomatic Julia version of the algorithm was written that is optimized to eliminate unnecessary memory allocations making heavy use of the Julia macro `@views`. This algorithm was submitted as a PR to [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl) and later merged.

# **Buckling**

After programming the LOBPCG algorithm to find generalized eigenvalues and testing it on dummy random matrices, it was time to test it on a real life problem. So I developed a linear elasticity analysis package called [LinearElasticity.jl](https://github.com/mohamed82008/LinearElasticity.jl) which can do stress and buckling analysis of linear elastic structures. Heavy use of [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) was made and [Makie.jl](https://github.com/JuliaPlots/Makie.jl) was used for visualization of the mode shapes. Results were then shared in the blog post [The Buckle Dance](https://mohamed82008.github.io/ScienceLounge/text/2018/07/11/The-buckle-dance/). The following figures show buckling modes computed of a compressed beam.

![First failure mode](https://user-images.githubusercontent.com/19524993/42571587-f60446c2-855a-11e8-8b95-eb8584917933.PNG)
![Second failure mode](https://user-images.githubusercontent.com/19524993/42571588-f63e91c4-855a-11e8-9507-8ae1e5e4d2c6.PNG)
![Third failure mode](https://user-images.githubusercontent.com/19524993/42571589-f6782682-855a-11e8-9b9a-39b4b078d999.PNG)

# **Preconditioners**

The third component of the project was to develop and combine a set of common preconditioners into a package which I called [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl). The package combines diagonal preconditioner, incomplete Cholesky preconditioner from the package [IncompleteSelectedInversion.jl](https://github.com/ettersi/IncompleteSelectedInversion.jl) and the algebraic multigrid (AMG) preconditioner from the package [AlgebraicMulrigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl), formerly known as `AMG.jl`.

## **Algebraic Multigrid**

### **Code validation**

In order to validate the AMG package, I read the majority of its lines of code and found a few bugs for unsymmetric matrices and unhandled corner cases which I fixed in [this PR](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/pull/43).

### **Code profiling**

I then profiled the memory and time requirements of different parts of the program to identify bottlenecks. Originally, the plan was to parallelize the splitting operator which determines coarse and fine variables, however this part was found to take only 2% of the operator building time and the majority of the time was spent in the so-called Galerkin projection that takes the form `A' B A` where `A` and `B` are sparse matrices. So I shifted my focus to making the AMG solver and preconditioner more lean by eliminating all allocations in the functions used during the solve process and *parallelizing* the smoother.

### **Code refactoring**

The `AlgebraicMultigrid.jl` package was written to closely map the similar Python package, [PyAMG](https://github.com/pyamg/pyamg). This naturally led to a lot of unnecessary memory allocations and other inefficiencies all over the place. So I decided to refactor the package to eliminate all unnecessary allocations during the call of the AMG operator. While building the operator still allocates plenty, this is only done once, so it was not my focus. The refactoring is part of the same AMG PR above.

### **Shared memory parallelism**

As part of my quest to learn and adapt Julia's experimental shared memory parallelism to make the AMG preconditioner faster, I had to play a lot with [KissThreading.jl](https://github.com/bkamins/KissThreading.jl). I found and fixed a few type instabilities, upgraded `KissThreading.jl` to v0.7 (merged PR), while maintaining [support for v0.6](https://github.com/bkamins/KissThreading.jl/pull/11). I also added a threaded batch `mapreduce` implementation which in v1.0 has comparable and often superior speed to that of the unthreaded `Base.mapreduce`. The main motivation for using `KissThreading.jl` over Julia's raw shared memory parallelism is the guaranteed type stability at the face of the infamous [closure bug](https://github.com/JuliaLang/julia/issues/15276). Working around this bug required significant code manipulation in `KissThreading.jl` which doesn't need to be repeated in other codes doing a similar pattern of parallelism, e.g. `map` or `mapreduce`. After sufficiently developing `KissThreading.jl`, I developed a multithreaded Jacobi smoother which can be found in [the same AMG PR](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/pull/43).

### **Disappointing experiments**

In order to test the efficacy of the AMG operator as a preconditioner, I conducted some experiments to perform stress and buckling analysis using `LinearElasticity.jl`, `IterativeSolvers.jl` and `Preconditioners.jl` and the results were not satisfactory. While there was a significant reduction in the number of iterations of these algorithms, this reduction was not enough to make up for the additional work done in the AMG operator's "smoother", which is called twice in every preconditioner call. This led the preconditioned conjugate gradient (CG) to do around 25% more matrix-vector multiplications' worth of work in total compared to the vanilla CG. For the LOBPCG algorithm, the results were similar but the gap was smaller. More improvements are due to make sure the AMG operator is actually effective in reducing the amount of computations required by CG or LOBPCG to solve their respective problems. This however is out of the scope of the GSoC project.

### **Dropping dependencies**

Upgrading AMG to Julia v1.0 was not easy since one of its dependencies, [JLD.jl](https://github.com/JuliaIO/JLD.jl), is not yet upgraded to support Julia v1.0. In order to replace it, I used native Julia `.jl` files instead, writing the contents of the `SparseMatrixCSC`s in code form by first producing a string representation of each of the fields of the `SparseMatrixCSC` in question then writing all the commands to produce the `SparseMatrixCSC` in the `.jl` file. My initial attempt led to loss in precision as Julia's string representation of a `Vector{Float64}` rounds it to 6 significant figures. After spending a while doubting everything else in the code, I finally found this cause of failed tests and fixed it by joining together the string representations of the individual `Float64`s using the `join` function in Julia.

# **Summary**

Overall, I found the GSoC project to be a lot fun and extremely beneficial in developing my software development skills. Needless to say, there were also a lot of challenges, surprises and occasional disappointments, but I think it's fair to say that this is all part of the deal. My overall impression is very positive. My mentor is a great guy named [Harmen Stoppels](http://stoppels.blog/) and I definitely recommend him for any future GSoC student interested in iterative solvers in Julia. If you are a future GSoC student interested in knowing more about my experience or if you find the work I did in this project interesting, please don't hesitate to contact me on the Julia Discourse, @mohamed82008.
