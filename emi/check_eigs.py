from emi.la import configure_cond_eps
from xii import ii_convert
from dolfin import info
import numpy as np

# Look at the conditioning number

GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'


def analyze_cond(problem, precond, ncases, alpha, get_cond, logfile):
    '''Study of module over ncases for fixed alpha'''
    # Annotate columns
    columns = ['ndofs', 'h', 'cond', 'flag']
    header = ' '.join(columns)
    
    # Stuff for command line printing as we go, eigenvalua bounds, cond number, nzeros
    formats = ['%d'] + ['%.2E'] + ['\033[1;37;34m%g(%g)\033[0m']
    msg = ' '.join(formats)

    mms_data = problem.setup_mms(alpha)

    case0 = 0
    with open(logfile, 'a') as stream:
        # Run in context manager to keep the data
        history = []
        for n in [4*2**i for i in range(case0, case0+ncases)]:
            # Setting up the problem means obtaining a block_mat, block_vec
            # and a list space or matrix, vector and function space
            try:
                AA, bb, W = problem.setup_problem(n, mms_data, alpha)
                Z = []
            except ValueError:
                AA, bb, W, Z = problem.setup_problem(n, mms_data, alpha)

            # Let's get the preconditioner as block operator
            try: 
                BB = precond(W, mms_data, alpha)
            except TypeError:
                try:
                    BB = precond(W, mms_data, alpha, AA)
                except TypeError:
                    BB = precond(W, mms_data, alpha, AA, Z)
                
            # Need a monolithic matrix
            A = ii_convert(AA)
            # spectrum expects matrices
            B = ii_convert(BB)   # ii_convert is identity for mat
            # Again monolithic kernel
            Z = list(map(ii_convert, Z)) if Z else Z
            # For dirichlet cases we might not have the list space
            h = W[0].mesh().hmin() if isinstance(W, list) else W.mesh().hmin()
            ndofs = A.size(0)

            cond, flag = get_cond(A, B, Z)
            # No convergence? 
            if cond is None: break
            # Write
            # Save current
            row = (ndofs, h, cond)
            stream.write('%d %g %.16f %s\n' % (ndofs, h, cond, flag))

            # Put the entire history because of eps monitor
            history.append(row)

            print('='*79)
            print(RED % str(alpha))
            print(GREEN % header)

            for i, row in enumerate(history):
                
                if len(history) > 1:
                    increment = history[i][-1] - history[i-1][-1]
                else:
                    increment = 0
                
                print(msg % (row + (increment, )))

            print('='*79)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from emi.utils import get_problem_parameters, split_jobs
    from emi.la import configure_cond_eps
    from dolfin import mpi_comm_world
    import argparse, os, importlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Which module to test?
    parser.add_argument('problem', type=str, help='Which problem to run')

    # We can have several preconditioners so which one ...
    parser.add_argument('-precond', type=int, default=0, 
                        help='Which preconditioner to use')

    # Number of mesh refinements to use in convergence study
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')

    # The range of problem parameters to check
    parser.add_argument('-alpha', type=float, nargs='+', default=[1.0],
                        help='Parameter values for problem setup')

    parser.add_argument('-save_dir', type=str, default='./results')
    parser.add_argument('-spawn', type=str, default='', help='rank/nproc for bash launched parameter sweeps')
    args, petsc_args = parser.parse_known_args()
    
    assert args.ncases > 0

    # NOTE: we pass parameters as -param_mu 0 1 2 3 -param_lmbda 1 2 3 4. Since until
    # the problem is know the number of param is not known they are not given to the
    # parser as arguments - we pick them up from petsc
    problem = args.problem
    if problem.endswith('.py'):
        problem = problem[:-3]
    problem = problem.replace('/', '.')
        
    module = importlib.import_module(problem)
    
    # What comes back is a geneterator over tensor product of parameter
    # ranges and cleaned up petsc arguments
    alphas, petsc_params = get_problem_parameters(petsc_args, module.PARAMETERS)

    # We log everyhing
    savedir = args.save_dir
    not os.path.exists(savedir) and os.mkdir(savedir)

    header = '\n'.join(['*'*60, '* %s with %s', '*'*60])

    # Get the preconditioner name base on choice        
    B = module.W_INNER_PRODUCTS[args.precond]
    Bname = B.__name__

    # Defaults - they here so they are recorded in the logfile
    my_params = {'-eps_tol': 1E-3, # eps cvrg tolerance
                 '-eps_max_it': 20000,      
                 '-eps_nev': 3,                      # How many
                 '-eps_monitor': 'none',
                 '-eps_type': 'krylovschur',
                 '-st_ksp_rtol': 1E-8, # cvrg tolerance ksp st
                 '-st_ksp_monitor_true_residual': 'none'}

    # Who configures
    configure_cond = configure_cond_eps

    # Set defaults
    for k in my_params:
        if k not in petsc_params:
            petsc_params[k] = my_params[k]
                
    # Fianlly configure
    get_cond = configure_cond(petsc_params)

    # So all the command line options
    cmd_options = args.__dict__.copy()
    cmd_options.update(petsc_params)  # Remember all options
    del cmd_options['spawn']
    cmd_options = '# %s\n' % (', '.join(map(str, list(cmd_options.items()))))

    my_jobs = split_jobs(args.spawn, alphas)
    n_jobs = len(my_jobs)
    for job_id, alpha in enumerate(my_jobs):
        # Encode the name of the current parameter
        alpha_str = '_'.join(['%s%g' % (p, getattr(alpha, p)) for p in module.PARAMETERS])

        logfile = os.path.join(savedir,
                               'eigs_%s_%s_%s.txt' % (problem, Bname, alpha_str))
        with open(logfile, 'w') as stream:
            stream.write(cmd_options)  # Options and alpha go as comments
            stream.write('# %s\n' % alpha_str)

        print(RED % (('%d/%d' % (job_id, n_jobs)) + (header % (alpha_str, Bname))))

        # The analysis
        analyze_cond(module, B, args.ncases, alpha, get_cond, logfile)
