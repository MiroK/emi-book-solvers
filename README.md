# Iterative solvers for cell-based EMI models

This repository contains source codes and an environment used to produce
results of the chapter in the EMI book **EMI: CELL BASED MATHEMATICAL MODEL OF EXCITABLE CELLS**
concerning iterative solvers.

## Installation
### Obtaining the image
The environment is provided as a docker container built on top of FEniCS
2017.2.0 (which uses Python 2). The container can be built locally by

```
docker build --no-cache -t emi_book_solvers .
```
from the root directory of this repository. You would then run the container
as

```
docker run -it -v $(pwd):/home/fenics/shared emi_book_solvers
```

Alternatively, pull and run the image built in the cloud

```
docker run -it -v $(pwd):/home/fenics/shared mirok/emi-book-solvers
```

### Testing
Having lauched the container navigate to the source folder and launch
the test shell script

```
# You are inside docker shell
# fenics@268200ea18c2:~$ pwd
# /home/fenics

cd emi-book-solvers
cd emi
sh test.sh
```

If `You are all set message appears` everything is good to go.

## Monolithic solver experiments
Results of the chapter can be obtained by running convergence studies of
the different formulations. Below the single-dimensional primal formulation
is used. Boundedness of the Krylov iterations using different preconditioners
and different material parameters in the formulation can be tested with

```
fenics@3fb6d19d16c1:~/emi-book-solvers/emi$ python check_iters.py fem/prime_single.py -ncases 4 \
-param_eps 0.01 -param_kappa 1 -ksp_type cg -ksp_rtol 1E-10 -precond 2
```

Here 4 refinements, conductivity value 1, and time step parameter 0.01 are used. In addition,
the iterative solver is set to conjugate gradients (CG) using relative tolerance of
1E-10 and preconditioner corresponding to key 2 in the `W_RIESZ_MAPS` of
`prime_single.py` [module](https://github.com/MiroK/emi-book-solvers/blob/master/emi/fem/prime_single.py#L231).
Note that the single-dimensional formulation is the only one of the 4 discussed which results in 
positive symmetric matrix. The other formulations lead to saddle point systems and as such, they 
should be solved with `-ksp_type minres` (Minimal Residual method) or other suitable solvers from [PETSc](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPType.html). 
A succesfull run of `check_iters.py` results in a result file `iters_fem.prime_single_monolithic_precond_standard_kappa1_eps0.01.txt`
(since the key 2 preconditioner is named __monolithic__, cf. the source code), which is
located in `/home/fenics/emi-book-solvers/emi/results`. The file contains details of
the refinement study in particular the columns of the number of unknowns (ndofs), different errors
as specified in the `setup_error_monitor` function of the tested formulation script.
Important for the boundedness, the number of Krylov iterations is specified in the
`niters` column. Columns `dt`, `Bdt` and `total` contain the run time of the Krylov
loop, setup of the preconditioner and total time spent solving the system respecively.

```
# ('plot', 0), ('ncases', 4), ('-ksp_monitor_true_residual', 'none'), ('run_mms', 1), ('-ksp_max_it', 1500), ('-ksp_type', 'cg'), ('precond', 2), ('-ksp_rtol', '1E-10'), ('save_dir', './results'), ('case0', 2), ('problem', 'fem/prime_single.py'), ('-ksp_atol', 1e-20), ('norm', 'standard')
# kappa1_eps0.01
# l ndofs h e[|u|_1] r[|u|_1] e[|u|_0] r[|u|_0] niters dt Bdt total |r|_2 cond dimW_0 dimW_1
```

Condition numbers of the preconditioned problems are tested by running 
```
fenics@3fb6d19d16c1:~/emi-book-solvers/emi$ python check_eigs.py fem/prime_single.py -ncases 4 \
-param_eps 0.01 -param_kappa 1 -precond 2
```

The script stores the simulation output in the file `eigs_fem.prime_single_monolithic_inner_product_kappa1_eps0.01.txt`
which resides in `/home/fenics/emi-book-solvers/emi/results`. The file's content is as follows

```
fenics@3fb6d19d16c1:~/emi-book-solvers/emi$ cat results/eigs_fem.prime_single_monolithic_inner_product_kappa1_eps0.01.txt 
# ('-st_ksp_rtol', 1e-08), ('-eps_monitor', 'none'), ('ncases', 4), ('-eps_max_it', 20000), ('-eps_type', 'krylovschur'), ('-eps_nev', 3), ('-ksp_type', 'cg'), ('precond', 2), ('-st_ksp_monitor_true_residual', 'none'), ('save_dir', './results'), ('alpha', [1.0]), ('problem', 'fem/prime_single.py'), ('-ksp_rtol', '1E-10'), ('-eps_tol', 0.001)
# kappa1_eps0.01
33 0.353553 1.0000000000000300 direct
97 0.176777 1.0000000000000338 direct
321 0.0883883 1.0000000000000386 direct
1153 0.0441942 1.0000000000001226 direct
```

We can indeed see that preconditioning the system $Ax=b$ with $A^{-1}$ results in unit
condition number, cf. the third column.

## Domain decomposition solver experiments
Results concerning Robin-Neumann domain decomposition solver can be reproduced by

```
fenics@3fb6d19d16c1:~/emi-book-solvers/emi$ python check_sanity.py fem/DD.py -ncases 4 -param_eps 1 -param_kappa 1 -param_tol 1E-3
```

Here the switch `-param_tol` specifies the convergence tolerance for the algorithm.
For each mesh resolution a file `emi_DD_n$N_tol0.001_eps1.txt` is produced (above, corresponding
to 4 refinements `N` would take values 4, 8, 16, 32) which contains the convergence
history. Therefore, the final iteration counts can be extracted from the file. Run
time of the solver can be found in the file `sanity_fem.DD_standard_kappa1_eps1_tol0.001.txt`
under column `total`. As before, the file is located in the result folter.

## Troubleshooting
Please use the GitHub issue tracker for reporting issues, discussing code
contributions or requesting assistance.

## Citing
This code is based on several other packages in addition to [FEniCS](https://fenicsproject.org/citing/). 

1. [**FEniCS_ii**](https://github.com/MiroK/fenics_ii) is used to perform assembly of the multiscale varitional forms
2. [**cbc.block**](https://bitbucket.org/fenics-apps/cbc.block/src/master/) is used to represent the discrete operators
3. [**hsmg**](https://github.com/MiroK/hsmg) provides fractional Laplacian preconditioners

These can be cited as

1. _Kuchta, Miroslav. "Assembly of multiscale linear PDE operators." arXiv preprint arXiv:1912.09319 (2019)._
2. _Mardal, Kent-Andre, and Joachim Berdal Haga. "Block preconditioning of systems of PDEs." Automated solution of differential equations by the finite element method. Springer, Berlin, Heidelberg, 2012. 643-655._
3a. _Bærland, Trygve, Miroslav Kuchta, and Kent-Andre Mardal. "Multigrid methods for discrete fractional sobolev spaces." SIAM Journal on Scientific Computing 41.2 (2019): A948-A972.
3b. _Kuchta, Miroslav, et al. "Preconditioners for saddle point systems with trace constraints coupling 2d and 1d domains." SIAM Journal on Scientific Computing 38.6 (2016): B962-B987._
