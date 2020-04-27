# HYPRE's AMS preconditioner for div-div/curl-curl problems in 2D

from dolfin import *
from petsc4py import PETSc
from block.block_base import block_base
import numpy as np
import ufl


def vec(x):
    return as_backend_type(x).vec()


def mat(A):
    return as_backend_type(A).mat()


class HypreAMS(block_base):
    '''AMG auxiliary space preconditioner for Hdiv(0) norm'''
    def __init__(self, form=None, bc=None, hdiv0=False, A=None, V=None):
        # No user matrix
        if A is None:
            if isinstance(form, ufl.Form):
                t, dt = form.arguments()
                V, = set((t.function_space(), dt.function_space()))

                assert V.ufl_element().family() == 'Raviart-Thomas'
                assert V.ufl_element().degree() == 1

                mesh = V.mesh()
                assert mesh.geometry().dim() == 2

                shape = V.ufl_element().value_shape()
                L = inner(Constant(np.zeros(shape)), TrialFunction(V))*dx
                A, _ = assemble_system(form, L, bc)
            # Form can also be a space
            else:
                V = form
                assert V.ufl_element().family() == 'Raviart-Thomas'
                assert V.ufl_element().degree() == 1
                
                mesh = V.mesh()
                assert mesh.geometry().dim() == 2

                sigma, tau = TrialFunction(V), TestFunction(V)
        
                a = inner(div(sigma), div(tau))*dx
                if not hdiv0:
                    a += inner(sigma, tau)*dx

                f = Constant(np.zeros(V.ufl_element().value_shape()))
                L = inner(tau, f)*dx

            A, _ = assemble_system(a, L, bc)
        else:
            assert V is not None

            mesh = V.mesh()
            
        # AMS setup
        Q = FunctionSpace(mesh, 'CG', 1)
        G = DiscreteOperators.build_gradient(V, Q)

        pc = PETSc.PC().create(mesh.mpi_comm().tompi4py())
        pc.setType('hypre')
        pc.setHYPREType('ams')

        # Attach gradient
        pc.setHYPREDiscreteGradient(mat(G))

        # Constant nullspace (in case not mass and bcs)
        constants = [vec(interpolate(c, V).vector())
                     for c in (Constant((1, 0)), Constant((0, 1)))]

        pc.setHYPRESetEdgeConstantVectors(*constants)

        # NOTE: term mass term is accounted for automatically by Hypre
        # unless pc.setPoissonBetaMatrix(None)
        if hdiv0: pc.setHYPRESetBetaPoissonMatrix(None)

        pc.setOperators(mat(A))
        # FIXME: some defaults
        pc.setFromOptions()
        pc.setUp()

        self.pc = pc
        self.A = A   # For creating vec

    def matvec(self, b):
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)

        if x.size() != b.size():
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.pc.apply(vec(b), vec(x))

        return x
