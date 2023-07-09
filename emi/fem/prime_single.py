from dolfin import *
from xii import *
import numpy as np
from emi.utils import matrix_fromHs, H1_norm, L2_norm
import emi.fem.common as common
from block.algebraic.petsc import LU, AMG


# Dropping the multiplier and still having a symmetric problem
def setup_problem(n, mms, params):
    '''Single-dimensional primal formulation'''
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
    V0 = FunctionSpace(outer_mesh, 'CG', 1)
    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    
    W = [V0, V1]

    u0, u1 = list(map(TrialFunction, W))
    v0, v1 = list(map(TestFunction, W))

    Tu0, Tv0 = (Trace(f, interface_mesh) for f in (u0, v0))
    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    # The line integral
    n = OuterNormal(interface_mesh, [0.5, 0.5])
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa, epsilon = list(map(Constant, (params.kappa, params.eps)))
    
    a = block_form(W, 2)

    a[0][0] = kappa*inner(grad(u0), grad(v0))*dx + (1./epsilon)*inner(Tu0, Tv0)*dx_
    a[0][1] = -(1./epsilon)*inner(Tu1, Tv0)*dx_

    a[1][0] = -(1./epsilon)*inner(Tu0, Tv1)*dx_
    a[1][1] = inner(grad(u1), grad(v1))*dx + (1./epsilon)*inner(Tu1, Tv1)*dx_

    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)
    L[0] = inner(f1, v0)*dx
    L[0] += sum((1./epsilon)*inner(gi, Tv0)*dx_(i) for i, gi in enumerate(gGamma, 1))
     
    # Iface contribution
    L[1] = inner(f, v1)*dx
    L[1] += -sum((1./epsilon)*inner(gi, Tv1)*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    A, b = list(map(ii_assemble, (a, L)))
    
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    V0_bcs = [DirichletBC(V0, gi, facet_f, i) for i, gi in enumerate(gBdry, 1)]
    bcs = [V0_bcs, []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    # Error of the solution ...
    exact = mms_data.solution
    subdomains = mms_data.subdomains[1]
    
    def get_error(wh, subdomains=subdomains, exact=exact, params=params):
        u1h, uh = wh

        sigma_exact, u_exact, p_exact, I_exact = exact

        return (sqrt(H1_norm(u_exact[0], u1h)**2 + H1_norm(u_exact[1], uh)**2),
                sqrt(L2_norm(u_exact[0], u1h)**2 + L2_norm(u_exact[1], uh)**2))
    
    error_types = ('|u|_1', '|u|_0')
    
    return get_error, error_types


def cannonical_inner_product(W, mms, params):
    '''Block diagonal preconditioner based on H1xH1 with standard norms'''
    V0, V1 = W
    kappa = Constant(params.kappa)
    epsilon = Constant(params.eps)
    
    outer_mesh = V0.mesh()
    inner_mesh = V1.mesh()
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
    dx_ = Measure('dx', domain=interface_mesh)

    # Outer has bcs
    u0, v0 = TrialFunction(V0), TestFunction(V0)
    Tu0, Tv0 = (Trace(f, interface_mesh) for f in (u0, v0))

    a0 = kappa*inner(grad(u0), grad(v0))*dx# + (1./epsilon)*inner(Tu0, Tv0)*dx_
    b0 = assemble(inner(Constant(0), v0)*dx)  # Auxiliary
    B0 = ii_convert(ii_assemble(a0))

    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    bcs = [DirichletBC(V0, Constant(0), facet_f, i) for i in range(1, 1 + len(subdomains))]
    B0, b = apply_bc(B0, b0, bcs)

    # Inner
    u1, v1 = TrialFunction(V1), TestFunction(V1)
    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))
    
    a1 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx #(1./epsilon)*inner(Tu1, Tv1)*dx_
    B1 = ii_convert(ii_assemble(a1))

    return block_diag_mat([B0, B1])


def wGamma_inner_product(W, mms, params):
    '''
    Block diagonal preconditioner based on H1xH1 with norms based on 
    trace perturbed Poisson problem.
    '''
    V0, V1 = W
    kappa = Constant(params.kappa)
    epsilon = Constant(params.eps)
    
    outer_mesh = V0.mesh()
    inner_mesh = V1.mesh()
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
    dx_ = Measure('dx', domain=interface_mesh)

    # Outer has bcs
    u0, v0 = TrialFunction(V0), TestFunction(V0)
    Tu0, Tv0 = (Trace(f, interface_mesh) for f in (u0, v0))

    a0 = kappa*inner(grad(u0), grad(v0))*dx + (1./epsilon)*inner(Tu0, Tv0)*dx_
    b0 = assemble(inner(Constant(0), v0)*dx)  # Auxiliary
    B0 = ii_convert(ii_assemble(a0))

    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    bcs = [DirichletBC(V0, Constant(0), facet_f, i) for i in range(1, 1 + len(subdomains))]
    B0, b = apply_bc(B0, b0, bcs)

    # Inner
    u1, v1 = TrialFunction(V1), TestFunction(V1)
    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))
    
    a1 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx + (1./epsilon)*inner(Tu1, Tv1)*dx_
    B1 = ii_convert(ii_assemble(a1))

    return block_diag_mat([B0, B1])


def cannonical_riesz_map(W, mms, params):
    '''Exact Inverse'''
    B = cannonical_inner_product(W, mms, params)
    
    return block_diag_mat([LU(B[0][0]), LU(B[1][1])])


def wGamma_riesz_map(W, mms, params):
    '''Exact Inverse'''
    B = wGamma_inner_product(W, mms, params)
    
    return block_diag_mat([LU(B[0][0]), LU(B[1][1])])


def monolithic_inner_product(W, mms, params, AA):
    '''System itself as it is SPD'''
    return ii_convert(AA)


def monolithic_precond(W, mms, params, AA):
    '''Invert system by AMG'''
    A = AMG(ii_convert(AA))#, parameters={'pc_hypre_boomeramg_max_iter': 4})

    R = ReductionOperator([2], W)

    return R.T*A*R


def monolithic_precond_LU(W, mms, params, AA):
    '''Invert system by LU'''
    A = LU(ii_convert(AA))#, parameters={'pc_hypre_boomeramg_max_iter': 4})

    R = ReductionOperator([2], W)

    return R.T*A*R


def cannonical_riesz_map_AMG(W, mms, params):
    '''AMG inverse'''
    B = cannonical_inner_product(W, mms, params)
    
    return block_diag_mat([AMG(B[0][0]), AMG(B[1][1])])

# --------------------------------------------------------------------

# The idea now that we refister the inner product so that from outside
# of the module they are accessible without referring to them by name
W_INNER_PRODUCTS = {0: cannonical_inner_product,
                    1: wGamma_inner_product,
                    2: monolithic_inner_product}

# And we do the same for preconditioners / riesz maps
W_RIESZ_MAPS = {0: cannonical_riesz_map,
                1: wGamma_riesz_map,
                2: monolithic_precond,
                3: cannonical_riesz_map_AMG,
                4: monolithic_precond_LU}

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
