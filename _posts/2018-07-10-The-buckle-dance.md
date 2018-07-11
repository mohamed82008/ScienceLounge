---
title:  "GSoC - The Buckle Dance"
date:   2018-07-11
categories: text
---

## **Buckling**

In the last blog post, I talked about the rally to get the LOBPCG PR out of the window. This naturally led to the next part of my project which involved using the LOBPCG algorithm to help with a real life application. You may remember from the second last blog post, how is it the "rise of the quadratic terms" in the nonlinear strain formula leads to a very important generalized eigenvalue problem, that if solved can tell us how stable a specific mechanical structure is towards the load applied.

### **Element Stress Stiffness Matrices**

Identifying how much load it takes for a structure to break apart is a non-obvious task when the structure has a complicated shape and/or the load is complicated. In order to do buckling analysis using finite elements, one has to identify the so-called element stress stiffness matrices. The code used to generate these matrices is the core of any buckling analysis simulator. The code used in the package [LinearElasticity.jl](https://github.com/mohamed82008/LinearElasticity.jl) is shown below. This code makes use of the efficient finite element analysis package written in pure Julia [JuAFEM.jl](https://github.com/KristofferC/JuAFEM.jl) as well as a nice convenience package [Einsum.jl](https://github.com/ahwillia/Einsum.jl) which enabled me to use the Einstein summation notation commonly used in textbooks to refer to the concepts used below.

```julia
Kσ_e .= 0
reinit!(cellvalues, cell)
celldofs!(global_dofs, dh, cellidx)
for q_point in 1:getnquadpoints(cellvalues)
    dΩ = getdetJdV(cellvalues, q_point)
    for d in 1:dim
        ψ_e[(d-1)*dim+1:d*dim, (d-1)*dim+1:d*dim] .= 0
    end
    for a in 1:n_basefuncs
        ∇ϕ = shape_gradient(cellvalues, q_point, a)
        _u = @view dofs[(@view global_dofs[dim*(a-1) + (1:dim)])]
        @einsum u[i,j] = _u[i]*∇ϕ[j]
        @einsum ϵ[i,j] = 1/2*(u[i,j] + u[j,i])
        @einsum σ[i,j] = E*ν/(1-ν^2)*δ[i,j]*ϵ[k,k] + E*ν*(1+ν)*ϵ[i,j]
        for d in 1:dim
            ψ_e[(d-1)*dim+1:d*dim, (d-1)*dim+1:d*dim] .+= σ
            G[(dim*(d-1)+1):(dim*d), (a-1)*dim+d] .= ∇ϕ
        end
    end
    Kσ_e .+= G'*ψ_e*G*dΩ
end
```

While it may be hard to understand the above code without prior knowledge of finite element analysis, JuAFEM.jl and buckling theory, all the code is really doing is integrating a matrix-valued function over an element using Gaussian quadrature integration. The matrix to be integrated is `G'*ψ_e*G` where `G` is a special layout of shape function derivatives that can only be appreciated when manually derived! I derived every line of the above code once before, and maybe one day I will share these derivations here, stay tuned or read a book on buckling analysis!

`ψ_e` is nothing but `kron(eye(3,3), σ)` where `σ` is known as the stress tensor inside the element, computed from the nodal displacements `_u` using the 3 lines in the above code which make use of the `@einsum` macro. These lines assume that the structure is made of a linearly elastic material with only 2 parameters defining its behaviour: 1) the Young's modulus `E`, and 2) Poisson's ratio `ν`.


### **The Buckle Dance**

![First failure mode](https://user-images.githubusercontent.com/19524993/42571587-f60446c2-855a-11e8-8b95-eb8584917933.PNG)
![Second failure mode](https://user-images.githubusercontent.com/19524993/42571588-f63e91c4-855a-11e8-9507-8ae1e5e4d2c6.PNG)
![Third failure mode](https://user-images.githubusercontent.com/19524993/42571589-f6782682-855a-11e8-9b9a-39b4b078d999.PNG)
![Fourth failure mode](https://user-images.githubusercontent.com/19524993/42571593-f6ad8c1e-855a-11e8-939e-0b9f022ff6c5.PNG)

The above are four different failure modes that can happen to a beam compressed from top with a uniform pressure load. It takes a higher load to enable the modes of failure with more bumps and so typically the first one is the one of primary interest, since we would like to avoid triggering any instability whatsoever.

## **Preconditioners.jl**

After completing the buckling analysis task, it was time to move on to a new interesting challenge! This led to [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl).

In most iterative solvers, generally one is trying to approximate a matrix inversion process by a sequence of more efficient operations since inverting a matrix is way too expensive to use for big problems. Algorithms like the conjugate gradient (CG) algorithm for solving symmetric positive definite (SPD) systems of equations and the LOBPCG algorithm for eigenvalues can benefit greatly from an operator **P** that approximates multiplying the inverse of a certain matrix **M**, i.e. inv(**M**), by a vector **v**. The better this operator approximates the inverse of the matrix, the faster the convergence of CG or LOBPCG can be. The matrix **M** to be (approximately) inverted in CG which tries to solve a linear system of equations **A** **x** = **b** is simply the matrix **A**. While in LOBPCG, which tries to solve the eigenvalue problem **A** **x** = λ**B** **x**, **M** is usually a shift σ along the pencil of the matrices **A** and **B**, i.e. **M** = **A** - σ **B**.

Common preconditioners involve ignoring "annoying" parts of the matrix **M** and inverting the rest. For example, the diagonal preconditioner zeroes out all the off-diagonal terms and inverts the remaining diagonal matrix, which is efficient to invert. The incomplete Cholesky decomposition runs a few iterations of the Cholesky decomposition procedure ignoring any elements beyond a certain distance from the diagonal, or where the original sparse matrix **M** had a structural zero. Inverting such thin trinagular matrices is then much more efficient than inverting the whole original matrix.

One other interesting preconditioner is the algebraic multigrid preconditioner (AMG) which relies on dimensionality reduction of the matrix using the so-called "strong" variables, which are variables associated with large coefficients, to represent themselves and the so-called "weak" variables, which are variables associated with smaller coefficients. A resitriction operator and a prolongation/interpolation operator are made to facilitate the dimensionality reduction and expansion commonly used in AMG preconditioners.

More specifically, the AMG preconditioner tries to approximately solve the system **M** **x** = **v** by performing a very simple so-called "smoothing" procedure that gets us closer to the solution, though we may still be very far. This smoothing yields a temporary solution **y**. The solution **y** is then corrected using a correction vector **e** obtained by solving the system **M** **e** = **r**, where **r** = **v** - **M** **y**. This new system is now again solved very approximately by a "smoothing" operator but this time on a lower resolution matrix. The lower resolution matrix only involves a subset of the variables of the previous matrix, with new connectivity (coefficients) to reflect the interaction of those variables through the eliminated/reduced variables. The correction vector **e** obtained will then need to be corrected again since it was only approximate. This involves a recursion, where each correction is made on a lower resolution matrix until the deepest level where the system of equations is small enough to be directly solved by decomposition.

Finally, we just roll up all these corrections interpolating them over the reduced variables as we jump back up each level eventually correcting the top level solution **y**. After each correction on each level and before climbing up, another smoothing step is often performed to correct any "easy" errors that were introduced by the approximate correction. This whole process is often known as the V-cycle as it involves a recursion to find an approximate correction to the top-level solution with pre- and post- smoothing at each level.

The more the levels, the more accurate the correction can be and the more of a standalone solver the AMG operator can be. But the AMG operator can also be used as an approximate inverse operator with few levels. This means that it can be used with procedures such as CG and LOBPCG to improve their convergence properties.

In this part of the GSoC project, I wrapped a number of preconditioners in the package [Preconditioners.jl](https://github.com/mohamed82008/Preconditioners.jl) making them easily usable with IterativeSolvers.jl for easy plug and play. Preconditioners wrapped include:
1) Diagonal preconditioner,
2) Incomplete Cholesky preconditioner from the package [IncompleteSelectedInversion.jl](https://github.com/ettersi/IncompleteSelectedInversion.jl), and
3) The Ruge-Stuben and smoothed aggregation AMG preconditioners from the package [AMG.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).

## **JuMP-dev Workshop**

In the last month, I was busy for over a week preparing, flying and presenting at the [Second Annual JuMP-dev Workshop, Bordeaux 2018 ](http://www.juliaopt.org/meetings/bordeaux2018/). The talks should be up on Youtube any time. This event was very nice for learning about the new [MathOptInterface.jl](https://github.com/JuliaOpt/MathOptInterface.jl) and other advances in the optimization ecosystem of Julia. I also met a number of interesting people whose fields of work largely overlap with what I am planning to do in the near future so I came back home with alot of ideas! Special thanks to the folks at the MIT Sloan School of Management who funded my travel and stay (though still not reimbursed as of this time :p).
