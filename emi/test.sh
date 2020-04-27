python check_iters.py fem/prime_single.py -ncases 1 -param_kappa 1 -param_eps 1E-2

FILE=/home/fenics/emi-book-solvers/emi/results/iters_fem.prime_single_cannonical_riesz_map_standard_kappa1_eps0.01.txt
if test -f "$FILE"; then
    echo "You are all set"
fi
