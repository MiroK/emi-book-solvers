# Here we use LU for subdomains
from dolfin import *
from xii import *
import numpy as np
from emi.utils import H1_norm
from dd_utils import get_P0_gradient, flux_continuity, get_diff
import emi.fem.common as common
from block.algebraic.petsc import LU
from time import time


def setup_problem(n, mms, params):
    '''Domain decomposition for EMI'''
    # This is here only to get the API OKAY
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)
    mms.normals.append(n)
    
    # Marking
    inside = ['(0.25-tol<x[0])', '(x[0] < 0.75+tol)', '(0.25-tol<x[1])', '(x[1] < 0.75+tol)']
    inside = CompiledSubDomain(' && '.join(inside), tol=1E-10)

    mesh_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    inside.mark(mesh_f, 1)

    inner_mesh = SubMesh(base_mesh, mesh_f, 1)  # Inside
    outer_mesh = SubMesh(base_mesh, mesh_f, 0)  # Ouside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
            
    # Spaces
    V0 = FunctionSpace(outer_mesh, 'CG', 1)
    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    W = [V0, V1]
    
    u0, u1 = map(TrialFunction, W)
    v0, v1 = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(u0, v0)*dx
    a[1][1] = inner(u1, v1)*dx
    L = block_form(W, 1)
    
    A, b = map(ii_assemble, (a, L))
    
    return A, b, W


setup_mms = common.setup_mms


# Dropping the multiplier and still having a symmetric problem
def dd_solve(n, mms, params, tol):
    '''Domain decomposition for Laplacian using LU'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)

    # Marking
    inside = ['(0.25-tol<x[0])', '(x[0] < 0.75+tol)', '(0.25-tol<x[1])', '(x[1] < 0.75+tol)']
    inside = CompiledSubDomain(' && '.join(inside), tol=1E-10)

    mesh_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    inside.mark(mesh_f, 1)

    inner_mesh = SubMesh(base_mesh, mesh_f, 1)  # Inside
    outer_mesh = SubMesh(base_mesh, mesh_f, 0)  # Ouside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')

    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]
            
    # Spaces
    V0 = FunctionSpace(outer_mesh, 'CG', 1)
    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    
    W = [V0, V1]

    u0, u1 = map(TrialFunction, W)
    v0, v1 = map(TestFunction, W)

    Tu0, Tv0 = (Trace(f, interface_mesh) for f in (u0, v0))
    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))

    u0h, u1h = map(Function, W)
    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    # The line integral
    n = OuterNormal(interface_mesh, [0.5, 0.5])
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    # Data
    kappa, epsilon = map(Constant, (params.kappa, params.eps))
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    Tu0h, Tu1h = Trace(u0h, interface_mesh), Trace(u1h, interface_mesh)
    # Only it has bcs
    V0_bcs = [DirichletBC(V0, gi, facet_f, i) for i, gi in enumerate(gBdry, 1)]

    grad_uh1 = Function(VectorFunctionSpace(inner_mesh, 'DG', 0))
    Tgrad_uh1 = Trace(grad_uh1, interface_mesh)
    
    a0 = kappa*inner(grad(u0), grad(v0))*dx  # + (1./epsilon)*inner(Tu0, Tv0)*dx_
    L0 = inner(f1, v0)*dx
    # L0 += sum((1./epsilon)*inner(gi, Tv0)*dx_(i) for i, gi in enumerate(gGamma, 1))    
    # L0 += (1./epsilon)*inner(Tu1h, Tv0)*dx_
    L0 += -inner(dot(Tgrad_uh1, n), Tv0)*dx_

    a1 = inner(grad(u1), grad(v1))*dx + (1./epsilon)*inner(Tu1, Tv1)*dx_
    L1 = inner(f, v1)*dx
    L1 += -sum((1./epsilon)*inner(gi, Tv1)*dx_(i) for i, gi in enumerate(gGamma, 1))
    L1 += (1./epsilon)*inner(Tu0h, Tv1)*dx_

    parameters = {'pc_hypre_boomeramg_max_iter': 4}
    A0_inv, A1_inv = None, None

    u0h_, u1h_ = Function(V0), Function(V1)
    solve_time, assemble_time = 0., 0.

    rel = lambda x, x0: sqrt(abs(assemble(inner(x-x0, x-x0)*dx)))/sqrt(abs(assemble(inner(x, x)*dx)))
    
    k = 0
    converged = False
    errors = []

    Q = FunctionSpace(interface_mesh, 'CG', 1)
    q, q0 = Function(Q), Function(Q)
    while not converged:
        k += 1
        u0h_.vector().set_local(u0h.vector().get_local())  # Outer
        u1h_.vector().set_local(u1h.vector().get_local())  # Inner
        q0.vector().set_local(q.vector().get_local())
        
        # Solve inner
        A1, b1 = map(ii_convert, map(ii_assemble, (a1, L1)))

        now = time()
        if A1_inv is None:
            # Proxy because ii_assemble is slover
            assemble(inner(grad(u1), grad(v1))*dx + inner(u1, v1)*ds)
            A1_inv = LU(A1, parameters=parameters)        
        u1h.vector()[:] = A1_inv*b1 #solve(A1, u1h.vector(), b1)
        dt1 = time() - now

        Tgrad_uh1.vector()[:] = get_P0_gradient(u1h).vector()
        # Solve outer
        A0, b0 = map(ii_convert, map(ii_assemble, (a0, L0)))
        A0, b0 = apply_bc(A0, b0, V0_bcs)
        # solve(A0, u0h.vector(), b0)

        now = time()
        if A0_inv is None:
            A0_inv = LU(A0, parameters=parameters)
            _, _ = assemble_system(inner(grad(u0), grad(v0))*dx, inner(Constant(0), v0)*dx, V0_bcs)
        u0h.vector()[:] = A0_inv*b0
        dt0 = time() - now

        solve_time += dt0 + dt1

        # This is just approx
        now = time()
        assemble_time += 0 #time() - now
        
        flux_error = flux_continuity(u1h, u0h, interface_mesh, n, kappa2=kappa)
        rel0 = rel(u0h, u0h_)
        rel1 = rel(u1h, u1h_)

        q.vector()[:] = get_diff(u0h, u1h).vector()
        rel_v = rel(q, q0)

        print k, '->', u1h.vector().norm('l2'), u0h.vector().norm('l2'), flux_error, rel0, rel1, tol, rel_v, (dt0, dt1, assemble_time)
        errors.append(rel_v)
        
        converged = errors[-1] < tol or k > 200
        
    return (u0h, u1h), (solve_time, assemble_time), (flux_error, k), errors


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    # Error of the solution ...
    exact = mms_data.solution
    subdomains = mms_data.subdomains[1]
    
    def get_error(wh, subdomains=subdomains, exact=exact, mms_data=mms_data, params=params, tol=params.tol):
        n = mms_data.normals.pop()
        wh_, dt, (flux_error, niters), errors = dd_solve(n, mms_data, params, tol)
        u1h, uh = wh_

        print '>>>', dt
        np.savetxt('emi_DD_LU_n%d_tol%g_eps%g.txt' % (n, tol, params.eps),
                   errors)
        
        sigma_exact, u_exact, p_exact, I_exact = exact

        return (sqrt(H1_norm(u_exact[0], u1h)**2 + H1_norm(u_exact[1], uh)**2),
                sum(dt),
                flux_error, niters)

    error_types = ('|u|_1', 'total', '|cont|', '|niters|')
    
    return get_error, error_types

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps', 'tol')
