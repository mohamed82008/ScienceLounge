---
title:  "GSoC - Divorcing that PR"
date:   2018-06-12
categories: text
---

In the last blog post, I talked about the LOBPCG algorithm and my GSoC project's first 2 weeks. The 3rd and 4th weeks of my GSoC project were a long rally to get the LOBPCG PR out of the window.

## Result and tracing

At the start of the 3rd week, I made a PR to IterativeSolvers.jl to finalize that week's milestone. Soon after, I had a discussion with my mentor, Harmen Stoppels, and we agreed to implement a result and tracing structs approach similar to that of `Optim.jl`. The idea is to make it possible to return all the useful information that may be required by the user out of a LOBPCG run without having to litter the `return` statement with too many outputs. Had I chosen to `return` many outputs, the user would have had to remember the order these outputs are following which would be unnecessary mental workload. These useful information include things like: the number of iterations, tolerance, which eigenvectors converged, etc.

Another limitation of the implementation then was that a very basic trace was used which only allowed the storing of residual norms. While residual norms are important info to trace, one might also be interested in the trace of Ritz values. To that end, I implemented a `LOBPCGState` to store important information of every iteration. These are then stacked together if the `log` keyword argument was set to `true`. If the `log` keyword argument is set to `false`, the `trace` field of the `LOBPCGResults` struct is set to an empty `LOBPCGTrace` to maintain type stability.

## Tests-based workflow

After adding the above features, I realized that not all the LOBPCG code was tested. So I sat down and wrote some more tests for the constraints, fixed some errors and tried to increase the coverage as much as possible. I must say this is the first time I pay close attention to rigorous test coverage of my programs. Usually I would just make sure my program is running in all cases using notebooks and call it a day. However to convince Harmen and other spectators that the functions are working correctly in all scenarios, I had to be systematic about my tests making sure the results on my machine are reproducible on Travis. Coming up with and getting used to a workflow for debugging the erroring and failing tests took me a while, but then I think I found a good workflow. The tests I made were using different number types, for generalized and simple eigenvalue problems, for single and multiple eigenvalues, with and without constraints, with and without preconditioners, etc. The cases were almost too many to try out sequentially in a notebook. The `@testset` macro which admits a `for` loop in the header was my friend though, enabling me to define different combinations of the above test cases using relatively few lines of code. However, if one test case wouldn't work and I try to fix it, re-running all the tests again to check if they pass or not would take a few minutes. This is unreasonable. So as soon as I found a test case erroring or failing, I would extract the relevant code from the `test/lobpcg.jl` file and start playing with it in a separate Jupyter notebook. With the help of `Revise.jl`, I can then productively fix the problem with the code/test. So my final workflow was as follows:

1. Use VSCode to develop the files/package.
2. Run and debug basic manual test cases using a Jupyter notebook with the help of `Revise.jl`. This ensures that basic functionality is working.
3. Use VSCode to develop rigorous tests with maximal code coverage.
4. Use the command line to run `test/runtests.jl`.
5. Isolate the first erorring/failing test in a Jupyter notebook.
6. Use Jupyter, `Revise.jl` and VSCode to debug and fix the code and/or test case.
7. Move on to the next erroring/failing test until all are passed.
8. Commit the desired changes with the help of VSCode. Worthy of mention is the `(Un)Stage Selected Range` option of VSCode that I often use to select specific lines to stage in a commit. This makes it easier to separate changes made in the same coding session by their context.

## Decoupling the block size and number of eigenvalues

After all tests had passed and the PR was ready to merge, Harmen made one valid point about the possibility of separating `nev` which is the number of eigenvalues/eigenvectors desired and the block size `k` of the algorithm. He suggested to make it possible to find the `k` of the `nev` eigenvectors at a time. This may be useful in cases where not all the Ritz vectors converge at a similar rate. It may therefore be more efficient to use a smaller block size `k < nev` sequentially to find all the `nev` eigenvectors, `k` at a time, than to simply have `k == nev`. Supporting this functionality required adding an efficient `Constraint` updating mechanism. While the main idea seemed easy, that is just finding `k` eigenvectors in every iteration and adding them to the constraints to find the next batch of eigenvectors, the devil was indeed in the details. Efficiently updating the `Constraint` and handling corner cases where `nev` is not a multiple of `k` as well creating and updating the `LOBPCGResults` struct were the main challenges. After some hours of struggling with this, it was finally ready to merge!
