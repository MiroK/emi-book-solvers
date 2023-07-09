from dolfin import *
from xii import *
import numpy as np
from emi.utils import (matrix_fromHs, H1_norm, L2_norm, broken_norm, Hs_norm,
                       mat_add)
import emi.fem.common as common
from block.algebraic.petsc import LU, AMG
from hsmg.hseig import HsNorm


def setup_problem(n, mms, params):
    '''Multi-dimensional primal formulation'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)
    # Marking
    inside = ['(0.25-tol<x[0])', '(x[0] < 0.75+tol)', '(0.25-tol<x[1])', '(x[1] < 0.75+tol)']
    inside = CompiledSubDomain(' && '.join(inside), tol=1E-10)

    mesh_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    inside.mark(mesh_f, 1)

    inner_mesh = SubMesh(base_mesh, mesh_f, 1)  # Inside
    outer_mesh = SubMesh(base_mesh, mesh_f, 0)  # Ouside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
            
    # Spaces
    V1 = FunctionSpace(outer_mesh, 'CG', 1)
    V = FunctionSpace(inner_mesh, 'CG', 1)
    Q = FunctionSpace(interface_mesh, 'CG', 1)
    
    W = [V1, V, Q]

    u1, u, p = list(map(TrialFunction, W))
    v1, v, q = list(map(TestFunction, W))

    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))
    Tu, Tv = (Trace(f, interface_mesh) for f in (u, v))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    # The line integral
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa, epsilon = list(map(Constant, (params.kappa, params.eps)))
    
    a = block_form(W, 2)

    a[0][0] = kappa*inner(grad(u1), grad(v1))*dx
    a[0][2] = inner(Tv1, p)*dx_

    a[1][1] = inner(grad(u), grad(v))*dx
    a[1][2] = -inner(Tv, p)*dx_

    a[2][0] = inner(Tu1, q)*dx_
    a[2][1] = -inner(Tu, q)*dx_
    a[2][2] = -epsilon*inner(p, q)*dx_

    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)
    L[0] = inner(f1, v1)*dx
    L[0] += sum(inner(hi, Tv1)*dx_(i) for i, hi in enumerate(hGamma, 1))
    
    # Iface contribution
    L[1] = inner(f, v)*dx
    L[2] = sum(inner(gi, q)*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    A, b = list(map(ii_assemble, (a, L)))

    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    V1_bcs = [DirichletBC(V1, gi, facet_f, i) for i, gi in enumerate(gBdry, 1)]
    bcs = [V1_bcs, [], []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    exact = mms_data.solution
    subdomains = mms_data.subdomains[1]
    
    def get_error(wh, subdomains=subdomains, exact=exact, mms=mms_data, params=params):
        u1h, uh, ph = wh

        sigma_exact, u_exact, p_exact, I_exact = exact
        
        return (sqrt(H1_norm(u_exact[0], u1h)**2 + H1_norm(u_exact[1], uh)**2),
                broken_norm(Hs_norm(-0.5), subdomains)(I_exact, ph))
    
    error_types = ('|u|_1', '|I|_{-1/2}')
    
    return get_error, error_types


def cannonical_inner_product(W, mms, params):
    '''H1 x H1 x H^{-1/2} cap L^2'''
    V1, V, Q = W
    kappa = Constant(params.kappa)

    # Outer, H1_0 
    u, v = TrialFunction(V1), TestFunction(V1)

    outer_mesh = V1.mesh()
    
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    bcs = [DirichletBC(V1, Constant(0), facet_f, i) for i in range(1, 1 + len(subdomains))]
    
    V1_norm, _ = assemble_system(inner(grad(u), grad(v))*dx,
                                 inner(Constant(0), v)*dx,
                                 bcs)

    # Inner
    u, v = TrialFunction(V), TestFunction(V)

    V_norm = assemble(inner(grad(u), grad(v))*dx + inner(u, v)*dx)

    # Multiplier on the boundary
    # Fractional part
    Hs = matrix_fromHs(HsNorm(Q, s=-0.5))

    # L2 part
    epsilon = Constant(params.eps)
    p, q = TrialFunction(Q), TestFunction(Q)
    m = epsilon*inner(p, q)*dx
    M = assemble(m)

    Q_norm = mat_add(Hs, M)

    return block_diag_mat([V1_norm, V_norm, Q_norm])


def cannonical_riesz_map(W, mms, params):
    '''Exact inverse'''
    V1, V, Q = W
    kappa = Constant(params.kappa)

    # Outer, H1_0 
    u, v = TrialFunction(V1), TestFunction(V1)

    outer_mesh = V1.mesh()
    
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    bcs = [DirichletBC(V1, Constant(0), facet_f, i) for i in range(1, 1 + len(subdomains))]
    
    V1_norm, _ = assemble_system(inner(grad(u), grad(v))*dx,
                                 inner(Constant(0), v)*dx,
                                 bcs)

    # Inner
    u, v = TrialFunction(V), TestFunction(V)

    V_norm = assemble(inner(grad(u), grad(v))*dx + inner(u, v)*dx)

    # Multiplier on the boundary
    # Fractional part
    Hs = HsNorm(Q, s=[(1, -0.5), (params.eps, 0.0)])
    Hs = Hs**-1
    # L2 part
    # epsilon = Constant(params.eps)
    # p, q = TrialFunction(Q), TestFunction(Q)
    # m = epsilon*inner(p, q)*dx
    # M = assemble(m)

    # Q_norm = mat_add(Hs, M)

    # return block_diag_mat([V1_norm, V_norm, Q_norm])

    return block_diag_mat([LU(V1_norm), LU(V_norm), Hs])


def cannonical_riesz_map_AMG(W, mms, params):
    '''AMG inverse'''
    V1, V, Q = W
    kappa = Constant(params.kappa)

    # Outer, H1_0 
    u, v = TrialFunction(V1), TestFunction(V1)

    outer_mesh = V1.mesh()
    
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    bcs = [DirichletBC(V1, Constant(0), facet_f, i) for i in range(1, 1 + len(subdomains))]
    
    V1_norm, _ = assemble_system(inner(grad(u), grad(v))*dx,
                                 inner(Constant(0), v)*dx,
                                 bcs)

    # Inner
    u, v = TrialFunction(V), TestFunction(V)

    V_norm = assemble(inner(grad(u), grad(v))*dx + inner(u, v)*dx)

    # Multiplier on the boundary
    # Fractional part
    Hs = HsNorm(Q, s=[(1, -0.5), (params.eps, 0.0)])
    Hs = Hs**-1
    # L2 part
    # epsilon = Constant(params.eps)
    # p, q = TrialFunction(Q), TestFunction(Q)
    # m = epsilon*inner(p, q)*dx
    # M = assemble(m)

    # Q_norm = mat_add(Hs, M)

    # return block_diag_mat([V1_norm, V_norm, Q_norm])

    return block_diag_mat([AMG(V1_norm), AMG(V_norm), Hs])

# --------------------------------------------------------------------

# The idea now that we refister the inner product so that from outside
# of the module they are accessible without referring to them by name
W_INNER_PRODUCTS = {0: cannonical_inner_product}

# And we do the same for preconditioners / riesz maps
W_RIESZ_MAPS = {0: cannonical_riesz_map,
                1: cannonical_riesz_map_AMG}

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
