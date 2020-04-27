from emi.la import configure_iters_ksp
from xii import ii_assemble
from dolfin import Timer
import numpy as np


GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'


def analyze_iters(problem, precond, cases, alpha, norm_type, iter_solve, logfile, run_mms):
    '''Convergence study of module over ncases for fixed alpha'''
    mms_data = problem.setup_mms(alpha)
    
    # Compat
    try:
        error_monitor, error_types = problem.setup_error_monitor(mms_data, alpha, norm_type)
    except TypeError:
        error_monitor, error_types = problem.setup_error_monitor(mms_data, alpha)

        # Annotate columns
    columns = ['l', 'ndofs', 'h'] + sum((['e[%s]' % t,'r[%s]' % t] for t in error_types), []) + ['niters', 'dt', 'Bdt', 'total', '|r|_2', 'cond']
    header = ' '.join(columns)
    
    # Stuff for command line printing as we go
    formats = ['%d', '%d', '%.2E'] + sum((['%.4E', '%.2f'] for _ in error_types), []) + ['\033[1;37;34m%d\033[0m'] + ['%.2f', '%.2f', '%.2f', '%.2E', '%.2f']
    msg = ' '.join(formats)
    # At this point don't know how many subspaces
    msg_has_subspaces = False

    case0, ncases = cases

    e0, h0, rate = None, None, None
    # Exec in context so that results not lost on crash
    with open(logfile, 'a') as stream:
        history = [] # range(ncases): #
        for level, n in enumerate([4*2**i for i in range(case0, case0+ncases)], 1):
            # Setting up the problem means obtaining a block_mat, block_vec
            # and a list space or matrix, vector and function space
            try:
                AA, bb, W = problem.setup_problem(n, mms_data, alpha)
                Z = None
            except ValueError:
                AA, bb, W, Z = problem.setup_problem(n, mms_data, alpha)

            # Preconditioner as block_diag operator or something that
            # can be used by petsc
            B_setup = Timer('B_setup')
            try: 
                BB = precond(W, mms_data, alpha)
            except TypeError:
                try:
                    BB = precond(W, mms_data, alpha, AA)
                except TypeError:
                    BB = precond(W, mms_data, alpha, AA, Z)
            B_setup = B_setup.stop()
        
            # iter_solve most handle type
            estim_cond, ksp_time, niters, residuals, wh = iter_solve(AA, bb, BB, W, Z)
            # We're getting back iiFunction
            
            W = wh.function_space()
            h = W[0].mesh().hmin()
            subspaces = [f.function_space().dim() for f in wh]
            ndofs = sum(subspaces)

            if run_mms(ndofs):  # Decision based on dof count
                error = np.fromiter(error_monitor(wh), dtype=float)
        
                if e0 is None:
                    rate = np.nan*np.ones_like(error)
                else:
                    rate = np.log(error/e0)/np.log(h/h0)
                h0, e0 = h, error
            else:
                error = rate = np.nan*np.ones(len(error_types))

            r_norm = residuals[-1]
            row = [level, ndofs, h] + list(sum(zip(error, rate), ())) + [niters, ksp_time, B_setup, ksp_time+B_setup, r_norm, estim_cond] + subspaces
            
            history.append(row)

            if not msg_has_subspaces:
                msg = ' '.join([msg] + ['%d']*len(subspaces))
                header = ' '.join([header] + ['dimW_%d' % i for i in range(len(subspaces))])
                # Make record of what the result colums are
                stream.write('# %s\n' % header)

            msg_has_subspaces = True

            # Record info on iterations
            stream.write('%s\n' % ' '.join(map(str, row)))
            # The entire residual history
            residuals = ' '.join(('%g' % r) for r in residuals)
            stream.write('#! %s \n' % residuals)

            print '='*79
            print RED % str(alpha)
            print GREEN % header
            for row in history:
                print msg % tuple(row)
            print '='*79
        data = np.array(history)
        h = data[:, 2]
        for col in [3 + i for i in range(0, len(error)+4, 2)]:
            e = data[:, col]
            print np.polyfit(np.log(h), np.log(e), deg=1)
    # For outside plotting
    return wh
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from emi.utils import get_problem_parameters, split_jobs
    import argparse, os, importlib
    from dolfin import File, mpi_comm_world, set_log_level, WARNING

    # set_log_level(WARNING)
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Which module to test?
    parser.add_argument('problem', type=str, help='Which problem to run')

    # Which (template) preconitioner to use
    parser.add_argument('-precond', type=int, default=0, help='Which preconditioner to use')

    parser.add_argument('-plot', type=int, default=0, help='Dump solutions as PVD')

    # Initial case
    parser.add_argument('-case0', type=int, default=2,
                        help='First resolution is 4*2**cases0')

    # Number of mesh refinements to use in convergence study
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')

    # Not standard is typically the one defined by preconditioner inner
    # product
    parser.add_argument('-norm', type=str, default='standard',
                        help='Norm to be used by error monitor')

    # Might not be of interest ...
    parser.add_argument('-run_mms', type=int, default=1,
                        help='Run convergence study?')
    
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
    precond = module.W_RIESZ_MAPS[args.precond]
    name = precond.__name__

    # Defaults - they here so they are recorded in the logfile
    my_params = {'-ksp_rtol': 1E-8, # eps cvrg tolerance
                 '-ksp_atol': 1E-20,
                 '-ksp_max_it': 1500,
                 '-ksp_type': 'minres', 
                 '-ksp_monitor_true_residual': 'none'}

    # Set defaults
    for k in my_params:
        if k not in petsc_params:
            petsc_params[k] = my_params[k]

    # So all the command line options
    cmd_options = args.__dict__.copy()
    cmd_options.update(petsc_params)  # Remember all options
    del cmd_options['spawn']
    cmd_options = '# %s\n' % (', '.join(map(str, cmd_options.items())))

    # Look at small problems
    run_mms = lambda ndofs, x=args.run_mms == 1: x and ndofs < 4E6

    get_iters = configure_iters_ksp(petsc_params)
    # Go over all parameter combinations (mpirun will use one cpu per
    # subset of the parameters)
    my_jobs = split_jobs(args.spawn, alphas)
    n_jobs = len(my_jobs)
    for job_id, alpha in enumerate(my_jobs):
        # Encode the name of the current parameter
        alpha_str = '_'.join(['%s%g' % (p, getattr(alpha, p)) for p in module.PARAMETERS])

        logfile = os.path.join(savedir,
                               'iters_%s_%s_%s_%s.txt' % (problem, name, args.norm, alpha_str))
        with open(logfile, 'w') as stream:
            stream.write(cmd_options)  # Options and alpha go as comments
            stream.write('# %s\n' % alpha_str)

        print RED % (('%d/%d' % (job_id, n_jobs)) + (header % (alpha_str, name)))

        # The analysis
        wh = analyze_iters(module, precond, [args.case0, args.ncases], alpha, args.norm, get_iters, logfile, run_mms)

        # Dump each component
        if args.plot:
            out = './plots/wh_%s_%s_%s' % (problem, name, alpha_str)
            out = '_'.join([out, '%dsub.pvd'])
            out = os.path.join(savedir, out)

            for i, whi in enumerate(wh):
                whi.rename('f', '0')
                File(out % i) << whi
