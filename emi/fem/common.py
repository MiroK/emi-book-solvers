from collections import namedtuple
from dolfin import *
import sympy as sp
import ulfy


MMSData = namedtuple('MMSData', ('solution', 'rhs', 'subdomains', 'normals'))


def setup_mms(params):
    '''
    Simple MMS problem for UnitSquareMesh. Return MMSData.
    '''
    mesh = UnitSquareMesh(mpi_comm_self(), 2, 2)  # Dummy

    V = FunctionSpace(mesh, 'CG', 2)
    S = FunctionSpace(mesh, 'DG', 0)
    # Define as function to allow ufly substition
    kappa, kappa1, eps = Function(S), Function(S), Function(S)

    u = Function(V)
    sigma = kappa*grad(u)
    # Outer normals of inner
    normals = map(Constant, [(-1, 0), (1, 0), (0, -1), (0, 1)])

    # Forcing for first
    f = -div(sigma)

    # Second and its forcing
    u1 = Function(V)
    sigma1 = kappa1*grad(u1)
    f1 = -div(sigma1)
    
    # On interface we have difference in us (restricted)
    gGamma_i = lambda n: u1 - u - eps*dot(sigma, n)
    hGamma_i = lambda n: -dot(sigma1, n) + dot(sigma, n)

    # Multiplier is u1| restricted!
           
    # What we want to substitute
    x, y, kappa1_, kappa_, eps_ = sp.symbols('x y kappa1 kappa eps')

    u_ = sp.sin(pi*(x + y))
    #u1_ = u_/kappa1_ + (x-0.25)**2*(x-0.75)**2*(y-0.25)**2*(y-0.75)**2  # u1 - u1 = 0 so simple!
    u1_ = u_/kappa1_ + sp.cos(pi*(x-0.25)*(x-0.75))*sp.cos(pi*(y-0.25)*(y-0.75))

    subs = {u: u_, u1: u1_, kappa: kappa_, kappa1: kappa1_, eps: eps_}

    as_expression = lambda f: ulfy.Expression(f, subs=subs, degree=4,
                                              kappa=1,
                                              kappa1=params.kappa,
                                              eps=params.eps)

    # Solutions
    u_exact = as_expression(u)
    sigma_exact = as_expression(sigma)

    u1_exact = as_expression(u1)
    sigma1_exact = as_expression(sigma1)

    # Mulltiplier is du on the boundary
    p_exact = as_expression(u1 - u)
    I_exact = [as_expression(dot(sigma, n)) for n in normals]  # current
    
    # Data
    f = as_expression(f)
    f1 = as_expression(f1)
    # Dirichle data for outer boundary (piecewise)
    gBdry = [u1_exact]*4
    # Temperature jump (piecewise)
    gGamma = [as_expression(gGamma_i(n)) for n in normals] 
    # Flux jump
    hGamma = [as_expression(hGamma_i(n)) for n in normals]
    
    rhs = (f1, f, gBdry, gGamma, hGamma)
    # Subdomains for the outerboundary then interfaces then extra/intracelluler
    inside = ['(0.25-tol<x[0])', '(x[0] < 0.75+tol)', '(0.25-tol<x[1])', '(x[1] < 0.75+tol)']
    inside = ' && '.join(inside)

    outside = '!(%s)' % inside
    
    subdomains = [('near(x[0], 0.0)', 'near(x[0], 1.0)', 'near(x[1], 0.0)', 'near(x[1], 1.0)'),
                  ('near(x[0], 0.25)', 'near(x[0], 0.75)', 'near(x[1], 0.25)', 'near(x[1], 0.75)'),
                  (outside, inside)]
    # Solutions for inside and outside
    return MMSData(solution=[[sigma1_exact, sigma_exact], [u1_exact, u_exact], p_exact, I_exact],
                   rhs=rhs,
                   subdomains=subdomains,
                   normals=[normals, normals])
