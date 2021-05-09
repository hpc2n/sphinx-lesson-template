Example: LU factorization
-------------------------

.. objectives::

 - Apply StarPU to a real algorithm.

.. challenge::

    Parallize the finely-blocked LU factorization algorithm using StarPU.
    
    See: :doc:`task-basics-lu`

.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 59-283
    
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>
        #include <starpu.h>

        extern double dnrm2_(int const *, double const *, int const *);

        extern void dtrmm_(char const *, char const *, char const *, char const *,
            int const *, int const *, double const *, double const *, int const *,
            double *, int const *);

        extern void dlacpy_(char const *, int const *, int const *, double const *,
            int const *, double *, int const *);

        extern double dlange_(char const *, int const *, int const *, double const *,
            int const *, double *);

        extern void dtrsm_(char const *, char const *, char const *, char const *,
            int const *, int const *, double const *, double const *, int const *,
            double *, int const *);

        extern void dgemm_(char const *, char const *, int const *, int const *,
            int const *, double const *, double const *, int const *, double const *,
            int const *, double const *, double *, int const *);

        double one = 1.0;
        double minus_one = -1.0;

        // returns the ceil of a / b
        int DIVCEIL(int a, int b)
        {
            return (a+b-1)/b;
        }

        // returns the minimum of a and b
        int MIN(int a, int b)
        {
            return a < b ? a : b;
        }

        // returns the maxinum of a and b
        int MAX(int a, int b)
        {
            return a > b ? a : b;
        }

        void simple_lu(int n, int ldA, double *A)
        {
            for (int i = 0; i < n; i++) {
                for (int j = i+1; j < n; j++) {
                    A[i*ldA+j] /= A[i*ldA+i];

                    for (int k = i+1; k < n; k++)
                        A[k*ldA+j] -= A[i*ldA+j] * A[k*ldA+i];
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

        //
        // Kernel implementations:
        //
        //  Each implementation must have the following prototype:
        //      void (*func)(void *buffers[], void *args);
        //
        //  The 'buffers' argument encapsulates the input and output buffers. The 'args'
        //  argument encapsulates optional static arguments. 
        //

        // a CPU implementation for the kernel that computes a small LU decomposition
        static void small_lu(void *buffers[], void *args)
        {
            // In this case the kernel has a single input and output buffer. The buffer
            // is accessible through a matrix interface.
            struct starpu_matrix_interface *A_i = 
                (struct starpu_matrix_interface *)buffers[0];

            // we can now extract the relevant information from the interface
            double *ptr = (double *) STARPU_MATRIX_GET_PTR(A_i); // pointer
            const int n = STARPU_MATRIX_GET_NX(A_i);             // matrix dimension
            const int ld = STARPU_MATRIX_GET_LD(A_i);            // leading dimension

            // The runtime system guarantees that the data resides in the device memory 
            // (main memory in this case). Thus, we can call the simple_lu function to 
            // perform the actual computations.
            simple_lu(n, ld, ptr);
        }

        // a CPU implementation for the kernel that performs a block row/column update
        static void rc_update(void *buffers[], void *args)
        {
            // The first four dtrsm arguments are passed as statix arguments. This
            // allows us to use the same codelet to perform the block row and block 
            // column updates.
            char side, uplo, transa, diag;
            starpu_codelet_unpack_args(args, &side, &uplo, &transa, &diag);

            // This time we have two buffers:
            //   0 = a small LU decomposition that corresponds to the diagonal block
            //   1 = current row/column block
            //
            // Note that we do not have define the interface explicitly.

            dtrsm_(&side, &uplo, &transa, &diag,
                (int *)&STARPU_MATRIX_GET_NX(buffers[1]),
                (int *)&STARPU_MATRIX_GET_NY(buffers[1]),
                &one,
                (double *)STARPU_MATRIX_GET_PTR(buffers[0]),
                (int *)&STARPU_MATRIX_GET_LD(buffers[0]),
                (double *)STARPU_MATRIX_GET_PTR(buffers[1]),
                (int *)&STARPU_MATRIX_GET_LD(buffers[1]));

        }

        // a CPU implementation for the kernel that performs a trailing matrix update
        static void trail_update(void *buffers[], void *args)
        {
            // This time we have three buffers:
            //  0 = corresponding column block
            //  1 = corresponding row block
            //  2 = current trailing matrix block

            dgemm_("No Transpose", "No Transpose", 
                (int *)&STARPU_MATRIX_GET_NX(buffers[2]),
                (int *)&STARPU_MATRIX_GET_NY(buffers[2]),
                (int *)&STARPU_MATRIX_GET_NY(buffers[0]),
                &minus_one,
                (double *)STARPU_MATRIX_GET_PTR(buffers[0]),
                (int *)&STARPU_MATRIX_GET_LD(buffers[0]),
                (double *)STARPU_MATRIX_GET_PTR(buffers[1]),
                (int *)&STARPU_MATRIX_GET_LD(buffers[1]),
                &one,
                (double *)STARPU_MATRIX_GET_PTR(buffers[2]),
                (int *)&STARPU_MATRIX_GET_LD(buffers[2]));
        }

        //
        // Codelets
        //
        //  A codelet encapsulates the various implementations of a computational
        //  kernel.
        //

        // a codelet that computes a small LU decomposition
        static struct starpu_codelet small_lu_cl = {
            .name = "small_lu",                 // codelet name
            .cpu_funcs = { small_lu },          // pointers to the CPU implementations
            .nbuffers = 1,                      // buffer count
            .modes = { STARPU_RW }              // buffer access modes (read-write)
        };

        // a codelet that that performs a block row/column update
        static struct starpu_codelet rc_update_cl = {
            .name = "rc_update",
            .cpu_funcs = { rc_update },
            .nbuffers = 2,
            .modes = { STARPU_R, STARPU_RW }    // read-only, read-write
        };

        // a codelet that performs a trailing matrix update
        static struct starpu_codelet trail_update_cl = {
            .name = "trail_update",
            .cpu_funcs = { trail_update },
            .nbuffers = 3,
            .modes = { STARPU_R, STARPU_R, STARPU_RW }
        };

        void blocked_lu(int block_size, int n, int ldA, double *A)
        {
            const int block_count = DIVCEIL(n, block_size);

            // initialize StarPU
            int ret = starpu_init(NULL);

            if (ret != 0)
                return;

            // Each buffer that is to be passed to a task must be encapsulated inside a
            // data handle. This means that we must allocate and fill an array that 
            // stores the block handles.

            starpu_data_handle_t **blocks = 
                malloc(block_count*sizeof(starpu_data_handle_t *));

            for (int i = 0; i < block_count; i++) {
                blocks[i] = malloc(block_count*sizeof(starpu_data_handle_t));

                for (int j = 0; j < block_count; j++) {
                    // each block is registered as a matrix 
                    starpu_matrix_data_register(
                        &blocks[i][j],                      // handle
                        STARPU_MAIN_RAM,                    // memory node
                        (uintptr_t)(A+(j*ldA+i)*block_size), // pointer
                        ldA,                                 // leading dimension
                        MIN(block_size, n-i*block_size),    // row count
                        MIN(block_size, n-j*block_size),    // column count
                        sizeof(double));                    // element size
                }
            }

            // go through the diagonal blocks
            for (int i = 0; i < block_count; i++) {

                // insert a task that processes the current diagonal block
                starpu_task_insert(
                    &small_lu_cl,       // codelet
                    STARPU_PRIORITY,    // the next argument specifies the priority 
                    STARPU_MAX_PRIO,    // priority
                    STARPU_RW,          // the next argument is a read-write handle
                    blocks[i][i],       // handle to the diagonal block
                    0);                 // a null pointer finalizes the call

                // insert tasks that process the blocks to the right of the current 
                // diagonal block
                for (int j = i+1; j < block_count; j++) {

                    // blocks[i][j] <- L1(blocks[i][i]) \ blocks[i][j]
                    starpu_task_insert(&rc_update_cl,
                        STARPU_PRIORITY, MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-j+i),
                        STARPU_VALUE,   // the next argument is a static argument
                        "Left",         // pointer to the static argument
                        sizeof(char),   // size of the static argument
                        STARPU_VALUE, "Lower", sizeof(char),
                        STARPU_VALUE, "No transpose", sizeof(char),
                        STARPU_VALUE, "Unit triangular", sizeof(char),
                        STARPU_R, blocks[i][i],
                        STARPU_RW, blocks[i][j], 0);
                }

                // insert tasks that process the blocks below the current diagonal block
                for (int j = i+1; j < block_count; j++) {

                    // blocks[j][i] <- U(blocks[i][i]) / blocks[j][i]
                    starpu_task_insert(&rc_update_cl,
                        STARPU_PRIORITY, MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-j+i),
                        STARPU_VALUE, "Right", sizeof(char),
                        STARPU_VALUE, "Upper", sizeof(char),
                        STARPU_VALUE, "No transpose", sizeof(char),
                        STARPU_VALUE, "Not unit triangular", sizeof(char),
                        STARPU_R, blocks[i][i],
                        STARPU_RW, blocks[j][i], 0);
                }

                // insert tasks that process the trailing matrix
                for (int ii = i+1; ii < block_count; ii++) {
                    for (int jj = i+1; jj < block_count; jj++) {

                        // blocks[ii][jj] <- 
                        //               blocks[ii][jj] - blocks[ii][i] * blocks[i][jj]
                        starpu_task_insert(&trail_update_cl,
                            STARPU_PRIORITY, MAX(
                                MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-ii+i),
                                MAX(STARPU_MIN_PRIO, STARPU_MAX_PRIO-jj+i)),
                            STARPU_R, blocks[ii][i],
                            STARPU_R, blocks[i][jj],
                            STARPU_RW, blocks[ii][jj], 0);
                    }
                }
            }

            // free allocated resources
            for (int i = 0; i < block_count; i++) {
                for (int j = 0; j < block_count; j++) {

                    // The data handles must be unregistered. The main thread waits
                    // until all related tasks have been completed and the data is
                    // copied back to its original location.
                    starpu_data_unregister(blocks[i][j]);
                }
                free(blocks[i]);
            }
            free(blocks);

            starpu_shutdown();
        }


        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

        // computes C <- L * U
        void mul_lu(int n, int lda, int ldb, double const *A, double *B)
        {
            // B <- U(A) = U
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i+1; j++)
                    B[i*ldb+j] = A[i*lda+j];
                for (int j = i+1; j < n; j++)
                    B[i*ldb+j] = 0.0;
            }

            // B <- L1(A) * B = L * U
            dtrmm_("Left", "Lower", "No Transpose", "Unit triangular",
                &n, &n, &one, A, &lda, B, &ldb);
        }

        int main(int argc, char **argv)
        {
            //
            // check arguments
            //

            if (argc != 3) {
                fprintf(stderr,
                    "[error] Incorrect arguments. Use %s (n) (block size)\n", argv[0]);
                return EXIT_FAILURE;
            }

            int n = atoi(argv[1]);
            if (n < 1)  {
                fprintf(stderr, "[error] Invalid matrix dimension.\n");
                return EXIT_FAILURE;
            }

            int block_size = atoi(argv[2]);
            if (block_size < 2)  {
                fprintf(stderr, "[error] Invalid block size.\n");
                return EXIT_FAILURE;
            }

            //
            // Initialize matrix A and store a duplicate to matrix B. Matrix C is for
            // validation.
            //

            srand(time(NULL));

            int ldA, ldB, ldC;
            ldA = ldB = ldC = DIVCEIL(n, 8)*8; // align to 64 bytes
            double *A = (double *) aligned_alloc(8, n*ldA*sizeof(double));
            double *B = (double *) aligned_alloc(8, n*ldB*sizeof(double));
            double *C = (double *) aligned_alloc(8, n*ldC*sizeof(double));

            if (A == NULL || B == NULL || C == NULL) {
                fprintf(stderr, "[error] Failed to allocate memory.\n");
                return EXIT_FAILURE;
            }

            // A <- random diagonally dominant matrix
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++)
                    A[i*ldA+j] = B[i*ldB+j] = 2.0*rand()/RAND_MAX - 1.0;
                A[i*ldA+i] = B[i*ldB+i] = 1.0*rand()/RAND_MAX + n;
            }

            //
            // compute
            //

            struct timespec ts_start;
            clock_gettime(CLOCK_MONOTONIC, &ts_start);

            // A <- (L,U)
            blocked_lu(block_size, n, ldA, A);

            struct timespec ts_stop;
            clock_gettime(CLOCK_MONOTONIC, &ts_stop);

            printf("Time = %f s\n",
                ts_stop.tv_sec - ts_start.tv_sec +
                1.0E-9*(ts_stop.tv_nsec - ts_start.tv_nsec));

            // C <- L * U
            mul_lu(n, ldA, ldC, A, C);

            //
            // validate
            //

            // C <- L * U - B
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    C[i*ldC+j] -= B[i*ldB+j];

            // compute || C ||_F / || B ||_F = || L * U - B ||_F  / || B ||_F
            double residual = dlange_("Frobenius", &n, &n, C, &ldC, NULL) /
                dlange_("Frobenius", &n, &n, B, &ldB, NULL);

            printf("Residual = %E\n", residual);

            int ret = EXIT_SUCCESS;
            if (1.0E-12 < residual) {
                fprintf(stderr, "The residual is too large.\n");
                ret = EXIT_FAILURE;
            }

            //
            // cleanup
            //

            free(A);
            free(B);
            free(C);

            return ret;
        }
