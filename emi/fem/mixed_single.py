from dolfin import *
from xii import *
import numpy as np
from emi.utils import Hdiv_norm, L2_norm, broken_norm
import emi.fem.common as common
from block.algebraic.petsc import LU
from emi.hypre_ams import HypreAMS


def setup_problem(n, mms, params):
    '''Single-dimensional mixed formulation'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)

    # Marking of intra/extra-cellular domains
    outside, inside = mms.subdomains[2]

    cell_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    CompiledSubDomain(outside, tol=1E-10).mark(cell_f, 0)  # Not needed
    CompiledSubDomain(inside, tol=1E-10).mark(cell_f, 1)

    # These are just auxiliary so that interface can be grabbed
    inner_mesh = SubMesh(base_mesh, cell_f, 1)  # Inside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
    
    # Spaces
    V1 = FunctionSpace(base_mesh, 'RT', 1)
    Q1 = FunctionSpace(base_mesh, 'DG', 0)

    W = [V1, Q1]

    sigma1, u1 = list(map(TrialFunction, W))
    tau1, v1 = list(map(TestFunction, W))

    # Hdiv trace should you normal (though orientation seems not important)
    n = OuterNormal(interface_mesh, [0.5, 0.5]) 

    Tsigma1, Ttau1 = (Trace(f, interface_mesh, '+', n) for f in (sigma1, tau1))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    dx = Measure('dx', domain=base_mesh, subdomain_data=cell_f)
    # The line integral
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa1, epsilon = list(map(Constant, (params.kappa, params.eps)))
    
    a = block_form(W, 2)
    a[0][0] = inner((1./kappa1)*sigma1, tau1)*dx(0) + inner(sigma1, tau1)*dx(1)
    a[0][0] += epsilon*inner(dot(Tsigma1, n), dot(Ttau1, n))*dx_
    a[0][1] = inner(div(tau1), u1)*dx
    a[1][0] = inner(div(sigma1), v1)*dx
    
    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)

    # Outer boundary contribution
    n1 = FacetNormal(base_mesh)
    # Piece by piece
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    ds = Measure('ds', domain=base_mesh, subdomain_data=facet_f)
    L[0] = sum(inner(gi, dot(Ttau1, n1))*ds(i) for i, gi in enumerate(gBdry, 1))
    # Iface contribution
    L[0] += -sum(inner(gi, dot(Ttau1, n))*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    L[1] = -inner(f1, v1)*dx(0) - inner(f, v1)*dx(1)

    A, b = list(map(ii_assemble, (a, L)))

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    exact = mms_data.solution
    ifaces = mms_data.subdomains[1]
    subdomains = mms_data.subdomains[2]

    def get_error(wh, subdomains=subdomains, ifaces=ifaces, exact=exact,
                  normals=mms_data.normals[0], params=params, mms=mms_data):
        sigmah, uh = wh
        sigma_exact, u_exact, p_exact, I_exact = exact
        
        return (broken_norm(Hdiv_norm, subdomains[:])(sigma_exact[:], sigmah),
                broken_norm(L2_norm, subdomains[:])(u_exact[:], uh))                

    error_types = ('|sigma|_div', '|u|_0')
    
    return get_error, error_types


def wGamma_inner_product(W, mms, params):
    '''Hdiv cap L2(gamma) x L^2'''
    V, Q = W
    kappa1 = Constant(params.kappa)

    # Hdiv norm with piecewise conductivities
    mesh = V.mesh()
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

    outside, inside = mms.subdomains[2]
    CompiledSubDomain(outside, tol=1E-10).mark(cell_f, 0)  # Not needed
    CompiledSubDomain(inside, tol=1E-10).mark(cell_f, 1)
    # These are just auxiliary so that interface can be grabbed
    inner_mesh = SubMesh(mesh, cell_f, 1)  # Inside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')

    n = OuterNormal(interface_mesh, [0.5, 0.5])
    # 0 is outside and that is where we have kappa
    sigma, tau = TrialFunction(V), TestFunction(V)
    Tsigma, Ttau = (Trace(f, interface_mesh, '+', n) for f in (sigma, tau))
    
    dX = Measure('dx', subdomain_data=cell_f)
    dx_ = Measure('dx', domain=interface_mesh)

    epsilon = Constant(params.eps)

    V_norm = ii_convert(ii_assemble(
        inner(sigma, tau)*dX(0) +
        inner(sigma, tau)*dX(1) +
        epsilon*inner(dot(Tsigma, n), dot(Ttau, n))*dx_ +
        inner(div(sigma), div(tau))*dX))

    p, q = TrialFunction(Q), TestFunction(Q)
    Q_norm_L2 = assemble(inner(p, q)*dx)
    
    return block_diag_mat([V_norm, Q_norm_L2])


def cannonical_inner_product(W, mms, params):
    '''Hdiv x L2'''
    V, Q = W
    kappa1 = Constant(params.kappa)

    # Hdiv norm with piecewise conductivities
    mesh = V.mesh()
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    # Mark
    [CompiledSubDomain(subd, tol=1E-10).mark(cell_f, tag)
     for tag, subd in enumerate(mms.subdomains[2])]
    # 0 is outside and that is where we have kappa
    sigma, tau = TrialFunction(V), TestFunction(V)
    dX = Measure('dx', subdomain_data=cell_f)

    V_norm = assemble(inner(sigma, tau)*dX(0) +
                      inner(sigma, tau)*dX(1) +
                      inner(div(sigma), div(tau))*dX)

    p, q = TrialFunction(Q), TestFunction(Q)
    Q_norm = assemble(inner(p, q)*dx)

    return block_diag_mat([V_norm, Q_norm])


def cannonical_riesz_map(W, mms, params):
    '''Approx Riesz map w.r.t to H1 x Hdiv x L2'''
    from block.algebraic.petsc import LU, AMG
    # from weak_bcs.hypre_ams import HypreAMS

    B = cannonical_inner_product(W, mms, params)
    
    return block_diag_mat([LU(B[0][0]), LU(B[1][1])])


def wGamma_riesz_map(W, mms, params):
    '''Exact inverse'''
    B = wGamma_inner_product(W, mms, params)
    
    return block_diag_mat([LU(B[0][0]), LU(B[1][1])])


def cannonical_riesz_map_AMG(W, mms, params):
    '''Hdiv AMG inverse'''
    B = cannonical_inner_product(W, mms, params)
    
    return block_diag_mat([HypreAMS(A=B[0][0], V=W[0]),
                           LU(B[1][1])])

# --------------------------------------------------------------------

# The idea now that we refister the inner product so that from outside
# of the module they are accessible without referring to them by name
W_INNER_PRODUCTS = {0: cannonical_inner_product,
                    1: wGamma_inner_product}

# And we do the same for preconditioners / riesz maps
W_RIESZ_MAPS = {0: cannonical_riesz_map,
                1: wGamma_riesz_map,
                2: cannonical_riesz_map_AMG}

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
