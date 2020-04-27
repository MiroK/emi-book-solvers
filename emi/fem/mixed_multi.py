from dolfin import *
from xii import *
import numpy as np
from emi.utils import Hdiv_norm, L2_norm, broken_norm, matrix_fromHs, mat_add
import emi.fem.common as common
from block.algebraic.petsc import LU
from emi.hypre_ams import HypreAMS
from hsmg.hseig import HsNorm


def setup_problem(n, mms, params):
    '''Multi-dimensional mixed formulation'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)

    # Marking of intra/extra-cellular domains
    outside, inside = mms.subdomains[2]

    cell_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    CompiledSubDomain(outside, tol=1E-10).mark(cell_f, 0)  # Not needed
    CompiledSubDomain(inside, tol=1E-10).mark(cell_f, 1)

    # These are just auxiliary so that interface can be grabbed
    inner_mesh = SubMesh(base_mesh, cell_f, 1)  # Inside
    outer_mesh = SubMesh(base_mesh, cell_f, 0)  # Ouside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
    
    # Spaces
    V1 = FunctionSpace(base_mesh, 'RT', 1)
    Q1 = FunctionSpace(base_mesh, 'DG', 0)
    Y = FunctionSpace(interface_mesh, 'DG', 0)

    W = [V1, Q1, Y]

    sigma1, u1, p = map(TrialFunction, W)
    tau1, v1, q = map(TestFunction, W)

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

    kappa1, epsilon = map(Constant, (params.kappa, params.eps))
    
    a = block_form(W, 2)
    a[0][0] = inner((1./kappa1)*sigma1, tau1)*dx(0) + inner(sigma1, tau1)*dx(1)
    a[0][1] = inner(div(tau1), u1)*dx
    a[1][0] = inner(div(sigma1), v1)*dx
    
    a[0][2] = inner(dot(Ttau1, n), p)*dx_
    a[2][0] = inner(dot(Tsigma1, n), q)*dx_
    a[2][2] = (-1./epsilon)*inner(p, q)*dx_

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
    
    L[1] = -inner(f1, v1)*dx(0) - inner(f, v1)*dx(1)
    # Iface contribution
    L[2] = -(1./epsilon)*sum(inner(gi, q)*dx_(i) for i, gi in enumerate(gGamma, 1))

    A, b = map(ii_assemble, (a, L))

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    # Error of the solution ...

    exact = mms_data.solution
    ifaces = mms_data.subdomains[1]
    subdomains = mms_data.subdomains[2]
    
    def get_error(wh, subdomains=subdomains, ifaces=ifaces, exact=exact, normals=mms_data.normals[0]):
        sigmah, uh, ph = wh
        print 'Mean ph', assemble(ph*dx)
        sigma_exact, u_exact, p_exact, I_exact = exact

        # Compute Hs norm on finer space
        Q = ph.function_space()
        Q = FunctionSpace(Q.mesh(), 'CG', 1)
        Hs = matrix_fromHs(HsNorm(Q, s=0.5))
        # Use interpolated
        ph = interpolate(ph, Q)
            
        return (broken_norm(Hdiv_norm, subdomains[:])(sigma_exact[:], sigmah),
                broken_norm(L2_norm, subdomains[:])(u_exact[:], uh),
                L2_norm(p_exact, ph))
    
    error_types = ('|sigma|_div', '|u|_0', '|v|_0')
    
    return get_error, error_types


def cannonical_inner_product(W, mms, params):
    '''Inner product of Hdiv x L2 x H^s'''
    V, Q, Y = W
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

    V_norm = assemble((1./kappa1)*inner(sigma, tau)*dX(0) +
                      inner(sigma, tau)*dX(1) +
                      inner(div(sigma), div(tau))*dX)

    p, q = TrialFunction(Q), TestFunction(Q)
    Q_norm = assemble(inner(p, q)*dx)

    # Multiplier on the boundary
    epsilon = Constant(params.eps)    
    # Fractional part
    Hs = matrix_fromHs(HsNorm(Y, s=0.5))  # Weight
    
    # L2 part
    p, q = TrialFunction(Y), TestFunction(Y)
    m = (1./epsilon)*inner(p, q)*dx
    M = assemble(m)

    Y_norm = mat_add(Hs, M)
    
    return block_diag_mat([V_norm, Q_norm, Y_norm])


def cannonical_riesz_map(W, mms, params, AA):
    '''Exact Riesz map w.r.t to Hdiv x L2 x H^s'''
    V, Q, Y = W
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

    V_norm = assemble((1./kappa1)*inner(sigma, tau)*dX(0) +
                      inner(sigma, tau)*dX(1) +
                      inner(div(sigma), div(tau))*dX)

    p, q = TrialFunction(Q), TestFunction(Q)
    Q_norm = assemble(inner(p, q)*dx)

    # Multiplier on the boundary
    epsilon = Constant(params.eps)    
    # Fractional part
    Hs = HsNorm(Y, s=[(1, 0.5), (1./params.eps, 0.)])  # Weight
    Hs = Hs**-1
    # L2 part
    #p, q = TrialFunction(Y), TestFunction(Y)
    #m = (1./epsilon)*inner(p, q)*dx
    #M = assemble(m)

    #Y_norm = mat_add(Hs, M)
    
    return block_diag_mat([LU(V_norm),
                           LU(Q_norm),
                           Hs])


def cannonical_riesz_map_AMG(W, mms, params, AA):
    '''Approx Riesz map w.r.t to Hdiv x L2 x H^s'''
    V, Q, Y = W
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

    V_norm = assemble((1./kappa1)*inner(sigma, tau)*dX(0) +
                      inner(sigma, tau)*dX(1) +
                      inner(div(sigma), div(tau))*dX)

    p, q = TrialFunction(Q), TestFunction(Q)
    Q_norm = assemble(inner(p, q)*dx)

    # Multiplier on the boundary
    epsilon = Constant(params.eps)    
    # Fractional part
    Hs = HsNorm(Y, s=[(1, 0.5), (1./params.eps, 0.)])  # Weight
    Hs = Hs**-1
    # L2 part
    #p, q = TrialFunction(Y), TestFunction(Y)
    #m = (1./epsilon)*inner(p, q)*dx
    #M = assemble(m)

    #Y_norm = mat_add(Hs, M)
    
    return block_diag_mat([HypreAMS(A=V_norm, V=V),
                           LU(Q_norm),
                           Hs])

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
