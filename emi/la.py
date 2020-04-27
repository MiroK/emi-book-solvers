# This is a collection of solvers for diagnostics of the systems
from dolfin import *
from xii import (ii_assemble, ii_convert, ii_Function, ii_PETScOperator,
                 ii_PETScPreconditioner, as_petsc_nest)
from emi.utils import serialize_mixed_space, randomize
from scipy.linalg import eigvalsh, eigh, eigvals, eig, svd
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import ufl


def my_eigvalsh(A, B, tol=1E-8, Z=None):
    '''Au = lmbda Bu transforming to EVP'''
    print 'Z is', Z
    if Z is None or not len(Z):
    # Transformation
        beta, U = np.linalg.eigh(B)
        print '\tDone power <<'
        Bnh = U.dot(np.diag(beta**-0.5).dot(U.T))
        # We have a complete set of vectors - now we'd like ignore those eigs
        # that belong to vectors parallel with Z
        S = Bnh.dot(A.dot(Bnh))

        return np.linalg.eigvalsh(S)

    # Full
    lmbda, V = my_eigh(A, B)
    # The idea is this; we have A U = M U Lmbda
    # Sinze Z.T*A*U must map to (k, n) matrix of zeros
    # Then Z.T*A*U must have in each row a value that will match up with
    # zero eigenvalue on the diagonal. Then we kick it out
    Z = np.array(Z).reshape((len(A), -1))
    idx = np.argmax(np.abs((Z.T).dot(B.dot(V))), axis=1)

    return np.delete(lmbda, idx)
    

def my_eigh(A, B):
    '''Au = lmbda Bu transforming to EVP'''
    # Transformation
    beta, U = np.linalg.eigh(B)
    print '\tDone power'
    Bnh = U.dot(np.diag(beta**-0.5).dot(U.T))
    
    S = Bnh.dot(A.dot(Bnh))
    lmbda, V = np.linalg.eigh(S)

    return lmbda, Bnh.dot(V)


def slepc_eigh(A, B, Z=None, is_hermitian=True):
    '''GEVP by SLEPc via lapack'''
    A, B = as_backend_type(A).mat(), as_backend_type(B).mat()

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A, B)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setType('lapack')
    print 'Z is', Z
    if Z is not None:
        print 'Set'
        Z = [as_backend_type(z).vec() for z in Z]
        E.setDeflationSpace(Z)

    # type is -eps_type
    E.solve()

    eigw = []
    for i in range(E.getConverged()):
        eigw.append(E.getEigenvalue(i).real)
    eigw = np.array(eigw)
    idx = np.argsort(eigw)
    eigw = eigw[idx]

    v = A.createVecRight()
    eigv = []
    for i in idx:
        E.getEigenvector(i, v)
        eigv.append(v.array.real.tolist())
        print v.dot(Z[0])
    eigv = np.array(eigv)

    return eigw, eigv.T


def slepc_eigvalsh(A, B, Z):
    '''Eigenvalues only'''
    lmbda, vec = slepc_eigh(A, B, Z)

    return lmbda


def is_hermitian(A, tol):
    x = A.create_vec()
    try:
        x.randomize()
    except AttributeError:
        x = PETScVector(PETSc.Vec().createWithArray(np.random.rand(x.local_size())))
        
    y = A*x
    try:
        return (y - A.T*x).norm() < tol 
    except TypeError:
        return (y - A.T*x).norm('l2') < tol


def spectrum(A, B, full=False, Z=None):
    '''Solve the generalized eigenvalue problem Ax = lmbda Bx'''
    info('Solving for %d eigenvalues' % A.size(0))
    is_symmetric = is_hermitian(A, tol=1E-8*A.size(0))
    
    # NOTE: eigvalsh relies on B being SPD which we do not check
    if full:
        print 'Symmetry', is_symmetric, as_backend_type(A).mat().isHermitian(), is_hermitian(A, 1E-6)
        print np.linalg.norm(A.array() - A.array().T)
        # Symmetry dispatch
        if not is_symmetric:
            assert False  # FIXME: allow only symmetry
            try:
                return eig(A.array(), B.array()), is_symmetric
            except AttributeError:
                return eig(A.array(), B.array()), is_symmetric
        #try:
        #    # return my_eigh(A, B.array()), is_symmetric
        #    return slepc_eigh(A, B), is_symmetric
        #except AttributeError:
        assert Z is None
        return my_eigh(A.array(), B.array()), is_symmetric

    if not is_symmetric:
        assert False  # FIXME: allow only symmetry
        x = eigvals(A.array(), B.array())
        return x, is_symmetric

    # This direct way doesn't seem to pick up the nullspace
    # return slepc_eigvalsh(A, B, Z), is_symmetric

    Z_arr = [zi.get_local() for zi in Z]
    return my_eigvalsh(A.array(), B.array(), Z=Z_arr), is_symmetric


def get_extreme_eigw(A, B, Z, params, which):
    '''Use SLEPC for fish for extremal eigenvalues of GEVP(A, B)'''
    is_symmetric = is_hermitian(A, tol=1E-8*A.size(0))
    assert is_symmetric
    # From monolithic matrices to petsc
    A, B = (as_backend_type(x).mat() for x in (A, B))

    # Wrap
    opts = PETSc.Options()
    for key, value in params.items():
        opts.setValue(key, None if value == 'none' else value)

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    if Z is not None:
        Z = [as_backend_type(z).vec() for z in Z]
        E.setDeflationSpace(Z)

    E.setOperators(A, B)
    # type is -eps_type
    if is_symmetric:
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    else:
        assert False
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        
    # Whch
    E.setWhichEigenpairs(which)

    # For 'small' system the shifted problems will use direct solver
    flag = 'slepc_direct'
    # Otherwise using shift and invert spectral transformation with zero shift
    # itertively
    # NOTE: these can ge configured from commandiline
    if A.size[0] > 2E6:
        flag = 'slepc_iter'
        
        ST = E.getST()
        ST.setType('sinvert')
        KSP = ST.getKSP()
        KSP.setType('cg')  # How to invert the B matrix
        PC = KSP.getPC()
        # NOTE: with these setting the cost is as high as direct
        PC.setType('lu')
        PC.setFactorSolverPackage('mumps')

        KSP.setFromOptions()
    E.setFromOptions()
    
    E.solve()

    its = E.getIterationNumber()
    nconv = E.getConverged()

    eigw = [E.getEigenvalue(i).real for i in range(nconv)]

    return eigw, flag


def configure_cond_eps(params):
    '''Configured EPS solver which returns the condition number of GEVP'''
    # NOTE: the solver returns also how it computed the conditioner number 
    def get_cond(A, B, Z, params=params):
        '''Get the condition number of GEVP with a B'''
        system_size = A.size(0)
        # Go to direct to save time
        if not Z and system_size < 8000:
            eigs, _ = spectrum(A, B, full=False, Z=Z)
            lmin, lmax = np.sort(np.abs(eigs))[[0, -1]]

            cond = lmax/lmin
            return (cond, 'direct')

        lmin, _ = get_extreme_eigw(A, B, Z, params, SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        lmax, flag = get_extreme_eigw(A, B, Z ,params, SLEPc.EPS.Which.LARGEST_MAGNITUDE)

        print '\tmin', lmin
        print '\tmax', lmax

        # Some failed to converge
        if not lmin or not lmax: return (None, '')
            
        cond = max(np.abs(lmax))/min(np.abs(lmin))

        return cond, flag

    return get_cond


def configure_iters_ksp(params):
    '''Configured KSP solver'''
    # NOTE: A, b, B are cbc.block
    def get_iters(A, b, B, W, Z=None, params=params):
        ''' Solution and iters A: W->W', B:W'->W, b\in W' '''
        ## AA and BB as block_mat
        ksp = PETSc.KSP().create()
        ksp.setConvergenceHistory()
        ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)

        compute_eigs = params['-ksp_type'] == 'cg'
        if compute_eigs:
            ksp.setComputeEigenvalues(1)
        else:
            ksp.setComputeEigenvalues(0)
            
        if isinstance(W, list):
            # The problem should always be symmetric
            y = A.create_vec()
            y = randomize(y)

            if params['-ksp_type'] == 'minres':
                Ay = A*y
            #    At_y = (A.T)*y
            #    assert (Ay - At_y).norm() < 1E-6*sum(yi.size() for yi in y), (Ay - At_y).norm() 

            ksp.setOperators(ii_PETScOperator(A, Z))  # Wrap block_mat
            ksp.setPC(ii_PETScPreconditioner(B, ksp))  # Wrapped block_op
        
            wh = ii_Function(W) 
            # Want the iterations to start from random
            wh.block_vec().randomize()
            # User configs
            opts = PETSc.Options()
            for key, value in params.iteritems():
                opts.setValue(key, None if value == 'none' else value)
                ksp.setFromOptions()
            
            Z is not None and Z.orthogonalize(wh.block_vec())

            # Time the solver 
            timer = Timer('solver')

            Z is not None and Z.orthogonalize(b)
            
            ksp.solve(as_petsc_nest(b), wh.petsc_vec())
            ksp_time = timer.stop()
            
        else:
            pc_config, Bmat = B
            # A for syste, B for preconditioner
            ksp.setOperators(as_backend_type(A).mat(), Bmat)
            # Now config
            pc = ksp.getPC()
            pc_options = pc_config(pc)

            params.update(pc_options)
            
            opts = PETSc.Options()
            # Database + user options
            for key, value in params.iteritems():
               opts.setValue(key, None if value == 'none' else value)
            # Apply
            pc.setFromOptions()
            ksp.setFromOptions()

            wh = Function(W)
            x = as_backend_type(wh.vector())
            x.set_local(np.random.rand(x.local_size()))

            timer = Timer('solver')
            ksp.solve(as_backend_type(b).vec(), x.vec())
            ksp_time = timer.stop()

            if isinstance(W.ufl_element(), (ufl.VectorElement, ufl.TensorElement)) or W.num_sub_spaces() == 1:
                wh = ii_Function([W], [wh])
            else:
                # Now get components
                Wblock = serialize_mixed_space(W)
                wh = wh.split(deepcopy=True)
    
                wh = ii_Function(Wblock, wh)
        # Done
        niters = ksp.getIterationNumber()
        residuals = ksp.getConvergenceHistory()

        if compute_eigs:
            eigs = np.abs(ksp.computeEigenvalues())
            cond = np.max(eigs)/np.min(eigs)
        else:
            cond = -1
        
        return cond, ksp_time, niters, residuals, wh

    return get_iters
