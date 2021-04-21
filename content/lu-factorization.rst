Example: LU factorization
-------------------------

.. objectives::

 - Apply task-based parallelism to a real algorithm.

Algorithm
^^^^^^^^^

The algorithm we are considering is called LU factorization (or LU decomposition) without pivoting.
Let :math:`A` be a square matrix. 
An LU factorization refers to the factorization of :math:`A` into lower triangular matrix :math:`L` and an upper triangular matrix :math:`U`:

.. math:: A = L U.

For example, consider the following example from Wikipedia:

.. math::
    
    \left[
    \begin{matrix}
    4 & 3 \\
    6 & 3
    \end{matrix}
    \right]
    =
    \left[
    \begin{matrix}
    1   & 0 \\
    1.5 & 1
    \end{matrix}
    \right]
    \left[
    \begin{matrix}
    4   & 3 \\
    0   & -1.5
    \end{matrix}
    \right].

If we want to solve a system of linear equations

.. math:: A x = b

for :math:`x` and have already computed the LU factorization

.. math:: A = A U,

then we can solve :math:`A x = b` as follows:

 1. Solve the equation :math:`L y = b` for :math:`y`.

 2. Solve the equation :math:`U x = y` for :math:`x`.

The factorization is typically normalized such that the matrix :math:`L` is unit lower triangular, i.e., the diagonal entries are all ones.
This allows us to store the factor matrices on top of the original matrix:

.. math::

    A \leftarrow U + L - I.

In case of the earlier example, we get
    
.. math::
    
    \left[
    \begin{matrix}
    4   & 3 \\
    0   & -1.5
    \end{matrix}
    \right]
    +
    \left[
    \begin{matrix}
    1   & 0 \\
    1.5 & 1
    \end{matrix}
    \right]
    -
    \left[
    \begin{matrix}
    1 & 0 \\
    0 & 1
    \end{matrix}
    \right]
    =
    \left[
    \begin{matrix}
    4 & 3 \\
    1.5 & -1.5
    \end{matrix}
    \right].

The scalar sequential version of the algorithm is rather simple:
    
.. code-block:: c
    :linenos:
    
    void simple_lu(int n, int ld, double *A)
    {
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                A[i*ld+j] /= A[i*ld+i];

                for (int k = i+1; k < n; k++)
                    A[k*ld+j] -= A[i*ld+j] * A[k*ld+i];
            }
        }
    }

The LU factorization of the :math:`n \times n` matrix :math:`A` (column-major order with leading dimension `ld`) is stored on top of the original matrix :math:`A`.

Parallel algorithm
^^^^^^^^^^^^^^^^^^

