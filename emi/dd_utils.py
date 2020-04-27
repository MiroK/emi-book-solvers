from dolfin import *
from xii import *

from xii.assembler.trace_matrix import trace_mat_no_restrict


def get_P0_gradient(f):
    '''P0 function that is the gradient of scalar function'''
    V = f.function_space()
    mesh = V.mesh()

    elm = V.ufl_element()
    assert elm.value_shape() == ()
    assert elm.family() == 'Lagrange'
    assert elm.degree() == 1

    Q = VectorFunctionSpace(mesh, 'DG', 0)
    q = TestFunction(Q)
    
    L = CellVolume(mesh)**-1*inner(grad(f), q)*dx
    b = assemble(L)
    # And the result
    p = Function(Q, b)

    return p


def flux_continuity(u1, u2, iface, n, kappa1=Constant(1), kappa2=Constant(1)):
    grad_u1 = get_P0_gradient(u1)
    grad_u2 = get_P0_gradient(u2)

    T1 = Trace(grad_u1, iface)
    T2 = Trace(grad_u2, iface)

    e = kappa1*dot(T1, n) - kappa2*dot(T2, n)
    return ii_assemble(inner(e, e)*dx(domain=iface))


def get_diff(u_ext, u_int):
    V = u_int.function_space()
    mesh = V.mesh()
    interface_mesh = BoundaryMesh(mesh, 'exterior')
        
    V = FunctionSpace(interface_mesh, 'CG', 1)
    Tu_ext = PETScMatrix(trace_mat_no_restrict(u_ext.function_space(), V))*u_ext.vector()
    Tu_int = PETScMatrix(trace_mat_no_restrict(u_int.function_space(), V))*u_int.vector()
        
    vh = Function(V, Tu_ext - Tu_int)
    return vh
