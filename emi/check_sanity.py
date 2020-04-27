# The idea here is to check that the system is assembled correctly so
# we run MMS with a direct
from emi.la import direct_solve
from dolfin import info, Timer
from xii import ii_convert
import numpy as np


GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'


def analyze(problem, cases, alpha, norm_type, logfile):
    '''Convergence study of module over ncases for fixed alpha'''
    mms_data = problem.setup_mms(alpha)  # Only depend on physical parameters

    # Compat
    try:
        error_monitor, error_types = problem.setup_error_monitor(mms_data, alpha, norm_type)
    except TypeError:
        error_monitor, error_types = problem.setup_error_monitor(mms_data, alpha)
    
    # Annotate columnts
    columns = ['ndofs', 'h'] + sum((['e[%s]' % t,'r[%s]' % t] for t in error_types), []) + ['subspaces']
    header = ' '.join(columns)
    print GREEN % header
    
    # Stuff for command line printing as we go
    formats = ['%d', '%.2E'] + sum((['%.4E', '\033[1;37;34m%.2f\033[0m'] for _ in error_types), [])
    msg = ' '.join(formats)

    # Make record of what the result colums are
    with open(logfile, 'a') as f: f.write('# %s\n' % header)

    msg_has_subspaces = False
    case0, ncases = cases
    
    e0, h0, rate = None, None, None
    # Exec in context so that results not lost on crash
    msg_history = []
    with open(logfile, 'a') as stream:
        for n in [4*2**i for i in range(case0, case0 + ncases)]:
            # Setting up the problem means obtaining a block_mat, block_vec
            # and a list space or matrix, vector and function space
            AA, bb, W = problem.setup_problem(n, mms_data, alpha)
            # Since direct solver expects monolithic
            info('\tConversion'); cvrt_timer = Timer('cvrt')
            A, b = map(ii_convert, (AA, bb))  # This is do nothing for monolithic

            # print np.sort(np.abs(np.linalg.eigvalsh(A.array())))
            info('\tDone (Conversion) %g' % cvrt_timer.stop())
            
            # wh = direct_solve(A, b, W, 'mumps')

            wh = direct_solve(A, b, W)  # Output is always iiFunction
            
            print [np.any(np.isnan(whi.vector().get_local())) for whi in wh]
            print [np.any(np.isinf(whi.vector().get_local())) for whi in wh]
            print [whi.vector().norm('l2') for whi in wh]
            
            # And later want list space
            W = wh.function_space()

            error = np.fromiter(error_monitor(wh), dtype=float)
            h = W[0].mesh().hmin()
            subspaces = [Wi.dim() for Wi in W]
            ndofs = sum(subspaces)
        
            if e0 is None:
                rate = np.nan*np.ones_like(error)
            else:
                rate = np.log(error/e0)/np.log(h/h0)
            h0, e0 = h, error

            # ndofs h zip of errors and rates
            row = [ndofs, h] + list(sum(zip(error, rate), ())) + subspaces

            if not msg_has_subspaces:
                msg = ' '.join([msg] + ['%d']*len(subspaces))
            msg_has_subspaces = True

            msg_history.append(row)
            print '='*79
            print RED % str(alpha)
            print GREEN % header
            for msg_row in msg_history:
                print msg % tuple(msg_row)
            print '='*79
                
            stream.write('%s\n' % ' '.join(map(str, row)))
    # Out for plotting
    return wh, mms_data


def collapse(thing):
    '''Consume all iterators'''
    for i in thing:
        try:
            i = iter(i)
            for j in i:
                yield j
        except:
            yield i
            
# --------------------------------------------------------------------

if __name__ == '__main__':
    from utils import get_problem_parameters, split_jobs
    import argparse, os, importlib
    from dolfin import (File, interpolate, Measure, inner, TrialFunction,
                        TestFunction, Function, solve, MeshFunction, CompiledSubDomain,
                        set_log_level, WARNING, FacetNormal, assemble, dot)

    set_log_level(WARNING)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Which module to test?
    parser.add_argument('problem', type=str, help='Which problem to run')

    # Number of mesh refinements to use in convergence study
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')

    parser.add_argument('-case0', type=int, default=0,
                        help='Run convergence study starting with')

    # Not standard is typically the one defined by preconditioner inner
    # product
    parser.add_argument('-norm', type=str, default='standard',
                        help='Norm to be used by error monitor')

    parser.add_argument('-save_dir', type=str, default='./results')
    parser.add_argument('-plot', type=int, default=0, help='Dump solutions as PVD')
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
    print args.spawn
    # We log everyhing
    savedir = args.save_dir
    not os.path.exists(savedir) and os.mkdir(savedir)
        
    # So all the command line options
    # Spawn is ignored because it differs across files
    options = {k: v for k, v in args.__dict__.items() if k != 'spawn'}
    options = '# %s\n' % (', '.join(map(str, options.items())))

    header = '\n'.join(['*'*60, '* %s', '*'*60])
    
    for alpha in split_jobs(args.spawn, alphas):
        # Encode the name of the current parameter
        alpha_str = '_'.join(['%s%g' % (p, getattr(alpha, p)) for p in module.PARAMETERS])

        logfile = os.path.join(savedir, 'sanity_%s_%s_%s.txt' % (problem, args.norm, alpha_str))

        with open(logfile, 'w') as f:
            f.write(options)  # Options and alpha go as comments
            f.write('# %s\n' % alpha_str)

        print RED % (header % str(alpha))

        wh, mms_data = analyze(module, (args.case0, args.ncases), alpha, args.norm, logfile)

        if args.plot:
            out = './plots/wh_%s_%s' % (problem, alpha_str)
            out = '_'.join([out, '%dsub.pvd'])
            out = os.path.join(savedir, out)

            eout = './plots/w_%s_%s' % (problem, alpha_str)
            eout = '_'.join([eout, '%dsub.pvd'])
            eout = os.path.join(savedir, eout)

            mesh = wh[0].function_space().mesh()
            facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
            CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
            CompiledSubDomain('near(x[1], 0)').mark(facet_f, 2)
            CompiledSubDomain('near(x[0], 1)').mark(facet_f, 3)
            CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)

            ds_ = Measure('ds', domain=mesh, subdomain_data=facet_f)
            n = FacetNormal(mesh)
            for i, (whi, whi_true) in enumerate(zip(wh, mms_data.solution)):
                whi.rename('f', '0')
                #whi.vector().zero()
                
                File(out % i) << whi
