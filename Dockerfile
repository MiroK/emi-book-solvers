FROM quay.io/fenicsproject/stable:2017.2.0

USER root

ENV PYTHONPATH=""

USER fenics

# xii dependency - which avoids scipy.linalg import eigh_tridiagonal
RUN git clone https://github.com/nschloe/quadpy.git && \
    cd quadpy && \
    git checkout v0.12.10 && \
    python setup.py install --user && \
    cd ..

RUN git clone https://github.com/MiroK/hsmg.git && \
    cd hsmg && \
    git checkout e7986135e0186c9853594a842c4a11c7b59f496b && \
    cd ..

ENV PYTHONPATH="/home/fenics/hsmg/":"$PYTHONPATH"

# Get fenics_ii
RUN git clone https://github.com/MiroK/fenics_ii.git && \
    cd fenics_ii && \
    git fetch --all  && \
    git checkout -b 49a18855d077ab6e63e4f0b4d6a2f061de7f36ba && \
    cd ..
    
ENV PYTHONPATH="/home/fenics/fenics_ii/":"$PYTHONPATH"

# cbc.block
RUN git clone https://github.com/MiroK/cbc.block.git && \
    cd cbc.block && \
    git checkout -b 747d6fd3775489a0fb4dee31c5906103bc5e3edf && \
    python setup.py install --user && \
    cd ..

# ulfy
RUN git clone https://github.com/MiroK/ulfy.git && \
    cd ulfy && \
    git checkout -b 42dfe51c821acffbccc0df26d7b9549a5cb949eb && \
    python setup.py install --user && \
    cd ..

# emi-boo-repo
RUN git clone https://github.com/MiroK/emi-book-solvers.git

ENV PYTHONPATH="/home/fenics/emi-book-solvers/":"$PYTHONPATH"

USER root
