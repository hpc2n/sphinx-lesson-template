StarPU runtime system (part 2)
------------------------------

.. objectives::

 - Understand more about StarPU
 - Understand how StarPU supports distributed memory
 - Understand how StarPU supports GPUs

Data handles and interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A task implementation **should not modify the task arguments** as these changes are not propagated to the other tasks.
Furthermore, the task arguments do not induce any task dependencies.
They are therefore only suitable for passing static arguments to the tasks.

**Data handles** are much more flexible as any modification made in one task are passed to the other tasks and these changes also induce task dependencies.
A data handle (`starpu_data_handle_t`) can encapsulate any conceivable data type.
However, the built-in data interfaces for scalars, vectors and matrices are adequate for many use cases:

.. code-block:: c
    
    void starpu_variable_data_register (starpu_data_handle_t *handle,
        int home_node,
        uintptr_t ptr,
        size_t size 
    )
    
    #define STARPU_VARIABLE_GET_PTR (interface)
    #define STARPU_VARIABLE_GET_ELEMSIZE (interface)

Above, `home_node` is the **memory node** where the variable is initially stored.
In most cases, the variable is initially stored in the main memory (`STARPU_MAIN_RAM`).
The argument `ptr` is a pointer to the variable (in the main memory) and the argument `size` is the size of the variable.
    
.. code-block:: c

    void starpu_vector_data_register (starpu_data_handle_t * handle,
        int home_node,
        uintptr_t ptr,
        uint32_t nx,
        size_t elemsize 
    )
    
    #define STARPU_VECTOR_GET_PTR (interface)
    #define STARPU_VECTOR_GET_NX (interface)
    #define STARPU_VECTOR_GET_ELEMSIZE (interface)

Above, the argument `nx` is the length of the vector and the argument `elemsize` is the size of a vector element.
    
.. code-block:: c

    void starpu_matrix_data_register (starpu_data_handle_t * handle,
        int home_node,
        uintptr_t ptr,
        uint32_t ld,
        uint32_t nx,
        uint32_t ny,
        size_t elemsize 
    )
    
    #define STARPU_MATRIX_GET_PTR (interface)
    #define STARPU_MATRIX_GET_NX (interface)
    #define STARPU_MATRIX_GET_NY (interface)
    #define STARPU_MATRIX_GET_LD (interface)
    #define STARPU_MATRIX_GET_ELEMSIZE (interface)

Above, the argument `ld` is the leading dimension of the matrix (row-major order), the argument `xn` is the width of the matrix and the argument `ny` is the height of the matrix.

For example, the following example allocates a matrix and **initializes** a data handle that encapsulates the matrix:

.. code-block:: c
    :linenos:
    :emphasize-lines: 2-4

    double *matrix = malloc(width * ld * sizeof(double));
    starpu_data_handle_t handle;
    starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
        (uintptr_t)matrix, ld, height, width, sizeof(double));

.. figure:: img/starpu_handles1.png

   Data handles.
        
The above example assumes that the matrix is stored in column-major order.

Each data handle must be **unregistered** before the main thread can access it again:

.. code-block:: c

    starpu_data_unregister(handle);
    
This blocks the main thread until all related tasks have been executed.

The easiest way to pass a data handle to a task is to declare it in the related codelet:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 18,19

    struct starpu_codelet
    {
        uint32_t where;
        int (*can_execute)(unsigned workerid, struct starpu_task *task, unsigned nimpl);
        enum starpu_codelet_type type;
        int max_parallelism;
        starpu_cpu_func_t cpu_func STARPU_DEPRECATED;
        starpu_cuda_func_t cuda_func STARPU_DEPRECATED;
        starpu_opencl_func_t opencl_func STARPU_DEPRECATED;
        starpu_cpu_func_t cpu_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_cuda_func_t cuda_funcs[STARPU_MAXIMPLEMENTATIONS];
        char cuda_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];
        char opencl_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_mic_func_t mic_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_mpi_ms_func_t mpi_ms_funcs[STARPU_MAXIMPLEMENTATIONS];
        const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];
        int nbuffers;
        enum starpu_data_access_mode modes[STARPU_NMAXBUFS];
        enum starpu_data_access_mode *dyn_modes;
        unsigned specific_nodes;
        int nodes[STARPU_NMAXBUFS];
        int *dyn_nodes;
        struct starpu_perfmodel *model;
        struct starpu_perfmodel *energy_model;
        unsigned long per_worker_stats[STARPU_NMAXWORKERS];
        const char *name;
        unsigned color;
        int flags;
        int checked;
    };

The `nbuffers` field stores the total number of data handles the task accepts and the `modes` field tabulates an access mode for each data handle.
The access mode can be one of the following:

:STARPU_NONE:               Not documented.
:STARPU_R:                  Read-only mode.
:STARPU_W:                  Write-only mode.
:STARPU_RW:                 Read-write mode. Equivalent to `STARPU_R | STARPU_W`.
:STARPU_SCRATCH:            Scratch buffer (one per device).
:STARPU_REDUX:              The data handle is used in a reduction-type operation.
:STARPU_COMMUTE:            Tasks can access this variable in an arbitrary order.
:STARPU_SSEND:              The data has to be sent using a synchronous and non-blocking mode (StarPU-MPI).
:STARPU_LOCALITY:           Tells the scheduler that the data handle is sensitive to data locality.
:STARPU_ACCESS_MODE_MAX:    Not documented.
    
Note that this limits the number of data handles passed to a task to `STARPU_NMAXBUFS`.
Furthermore, all tasks of a particular type must accept the **same number of data handles**.
The number of data handles passed to a codelet can be arbitrary but this feature is not covered during this course.

For example, the following example defines a codelet that accepts a single read-write data handle:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4-5

    struct starpu_codelet codelet =
    {
        .cpu_funcs = { func },
        .nbuffers = 1,
        .modes = { STARPU_RW }
    };

The data handles are passed to the `starpu_task_insert` function:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 3-4
    
    starpu_task_insert(
        &codelet,
        STARPU_RW,
        handle,
        0);

Finally, the task implementation extracts a matching **data interface** from the implementation arguments:

.. code-block:: c
    :linenos:
    :emphasize-lines: 3-4,6-9

    void func(void *buffers[], void *args)
    {
        struct starpu_matrix_interface *interface =
            (struct starpu_matrix_interface *)buffers[0];

        double *ptr = (double *) STARPU_MATRIX_GET_PTR(interface);
        int height = STARPU_MATRIX_GET_NX(interface);
        int width = STARPU_MATRIX_GET_NY(interface);
        int ld = STARPU_MATRIX_GET_LD(interface);

        process(height, width, ld, ptr);
    }


.. figure:: img/starpu_handles2.png

   Data interfaces. The pointers `matrix` and `ptr` do not necessarily point to the same memory location.

The runtime system guarantees that **data resides in the device memory** when a worker thread starts executing the task.
If necessary, StarPU copies the data from one memory space to another.
The scalar and vector data handles have their own interfaces: `starpu_variable_interface` and `starpu_vector_interface`.

If two tasks are given the same data handle in their argument lists, then an **implicit data dependency** may be induced between the tasks:

.. figure:: img/starpu_depedencies.png

   Two examples of data dependencies

.. challenge::

    Modify the example below as follows:
    
        1. Write a new task implementation (`add_cpu`) that 
        
            - accepts three data handles (variable / `int`) as arguments (`buffers[0]`, `buffers[1]` and `buffers[2]`),
            
            - extracts the data interfaces from `buffers`: `a_i`, `b_i` and `c_i`
        
            - adds up the first two arguments and stores the result to the third argument. 
        
        2. Write the corresponding codelet (`add_cl`).
        
            - Remember, the first two data handles are `STARPU_R` and the third `STARPU_W`.
        
        3. Create three integer variables (`int`): `a`, `b` and `c`. Initialize `b` to `7`.
        
        4. Register a data handle for each variable: `a_h`, `b_h` and `c_h`.
        
        5. Insert an `init_cl` task that initializes `a_h` to 10.
        
        6. Insert an `add_cl` task and give `a_h`, `b_h` and `c_h` as arguments.
        
        7. Unregister `a_h`, `b_h` and `c_h`.
        
        8. Print the variables `a`, `b` and `c`.

    .. code-block:: c
        :linenos:
    
        #include <stdio.h>
        #include <starpu.h>

        // a task implementation that initializes a variable to 10
        void init_cpu(void *buffers[], void *cl_arg)
        {
            struct starpu_variable_interface *a_i =
                (struct starpu_variable_interface *) buffers[0];
            int *a = (int *) STARPU_VARIABLE_GET_PTR(a_i);
            *a = 10;
        }

        // a task implementation that adds two numbers and return the sum
        struct starpu_codelet init_cl = {
            .cpu_funcs = { init_cpu },
            .nbuffers = 1,
            .modes = { STARPU_W }
        };

        int main()
        {
            int a;

            if (starpu_init(NULL) != 0)
                printf("Failed to initialize Starpu.\n");

            // initialize all data handles
            starpu_data_handle_t a_h;
            starpu_variable_data_register(
                &a_h, STARPU_MAIN_RAM, (uintptr_t)&a, sizeof(a));
            
            // insert tasks
            starpu_task_insert(&init_cl, STARPU_W, a_h, 0);

            // unregister all data handles
            starpu_data_unregister(a_h);

            printf("%d\n", a);

            starpu_shutdown();

            return 0;
        }
   
.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 13-29,38-43,47,53,56-59,63,67-68,70
    
        #include <stdio.h>
        #include <starpu.h>

        // a task implementation that initializes a variable to 10
        void init_cpu(void *buffers[], void *cl_arg)
        {
            struct starpu_variable_interface *a_i =
                (struct starpu_variable_interface *) buffers[0];
            int *a = (int *) STARPU_VARIABLE_GET_PTR(a_i);
            *a = 10;
        }

        // a task implementation that adds two numbers and returns the sum
        void add_cpu(void *buffers[], void *cl_arg)
        {
            struct starpu_variable_interface *a_i =
                (struct starpu_variable_interface *) buffers[0];
            int *a = (int *) STARPU_VARIABLE_GET_PTR(a_i);

            struct starpu_variable_interface *b_i =
                (struct starpu_variable_interface *) buffers[1];
            int *b = (int *) STARPU_VARIABLE_GET_PTR(b_i);

            struct starpu_variable_interface *c_i =
                (struct starpu_variable_interface *) buffers[2];
            int *c = (int *) STARPU_VARIABLE_GET_PTR(c_i);

            *c = *a + *b;
        }

        // initialization codelet
        struct starpu_codelet init_cl = {
            .cpu_funcs = { init_cpu },
            .nbuffers = 1,
            .modes = { STARPU_W }
        };

        // addition codelet
        struct starpu_codelet add_cl = {
            .cpu_funcs = { add_cpu },
            .nbuffers = 3,
            .modes = { STARPU_R, STARPU_R, STARPU_W }
        };

        int main()
        {
            int a, b = 7, c;

            if (starpu_init(NULL) != 0)
                printf("Failed to initialize Starpu.\n");

            // initialize all data handles
            starpu_data_handle_t a_h, b_h, c_h;
            starpu_variable_data_register(
                &a_h, STARPU_MAIN_RAM, (uintptr_t)&a, sizeof(a));
            starpu_variable_data_register(
                &b_h, STARPU_MAIN_RAM, (uintptr_t)&b, sizeof(b));
            starpu_variable_data_register(
                &c_h, STARPU_MAIN_RAM, (uintptr_t)&c, sizeof(c));

            // insert tasks
            starpu_task_insert(&init_cl, STARPU_W, a_h, 0);
            starpu_task_insert(&add_cl, STARPU_R, a_h, STARPU_R, b_h, STARPU_W, c_h, 0);

            // unregister all data handles
            starpu_data_unregister(a_h);
            starpu_data_unregister(b_h);
            starpu_data_unregister(c_h);

            printf("%d + %d = %d\n", a, b, c);

            starpu_shutdown();

            return 0;
        }
    
    .. code-block:: bash
    
        $ gcc -o starpu_program starpu_program.c -Wall -lstarpu-1.3
        $ ./starpu_program 
        10 + 7 = 17

Distributed memory
^^^^^^^^^^^^^^^^^^

StarPU supports distributed memory through MPI in three different ways:

 1. Without StarPU-MPI. 
 
     - A programmer must manually transfer the data between StarPU data handles and MPI.
     - Not generally recommended but might be a good stopgap solution. 
 
 2. With StarPU-MPI.
 
     - A programmer replaces the :code:`MPI_Recv()` and :code:`MPI_Send()` calls with :code:`starpu_mpi_irecv_detached()` and :code:`starpu_mpi_isend_detached()` calls.
       These functions act directly on the StarPU data handles.
     
 3. With MPI Insert Task Utility.
 
     - A programmer replaces the :code:`starpu_task_insert()` calls with :code:`starpu_mpi_task_insert()` calls.
       In addition, one must use the :code:`starpu_mpi_data_register()` function to tell which MPI process owns each data handle.

The second and third approach allocate one CPU core for MPI communications.
In the third approach, the :code:`starpu_mpi_task_insert()` function takes into account the task dependencies and the data distribution, and **generates the necessary communication pattern automatically**:

.. figure:: img/mpi.png

In the above illustration, each MPI process has a copy of the entire task graph. 
Two things can happen:

 1. If the MPI process is going to execute a task, it can **receive** any missing data handle from the MPI process that owns the data handle.
 2. If the MPI process is not going to execute a task, it will **send** any data handles it owns to the MPI process that is going to execute the task.
 
This all happens automatically and asynchronously. 
The task implementations do not require any modifications!
StarPU-MPI also implements a MPI cache that caches data handles that were not modified.

Consider the following example where each MPI process writes its rank to a data handle and then passes it to the neighbouring MPI process:

.. code-block:: c
    :linenos:
    :emphasize-lines: 48-50,63-64,69-72,76-79,86
    
    #include <stdio.h>
    #include <starpu.h>
    #include <starpu_mpi.h>

    // a codelet that initializes a data handle with MPI process' world rank

    void write_number_cpu(void *buffers[], void *cl_arg)
    {
        int world_rank = starpu_mpi_world_rank();
        int *value = (int *) STARPU_VARIABLE_GET_PTR(buffers[0]);
        
        *value = world_rank;
        printf("Rank %d writes value %d.\n", world_rank, *value);
    }

    struct starpu_codelet write_number_cl = {
        .cpu_funcs = { write_number_cpu },
        .nbuffers = 1,
        .modes = { STARPU_W }
    };

    // a codelet that prints the contents of a data handle

    void read_number_cpu(void *buffers[], void *cl_arg)
    {
        int world_rank = starpu_mpi_world_rank();
        int value = *((int *) STARPU_VARIABLE_GET_PTR(buffers[0]));
        printf("Rank %d reads value %d.\n", world_rank, value);
    }

    struct starpu_codelet read_number_cl = {
        .cpu_funcs = { read_number_cpu },
        .nbuffers = 1,
        .modes = { STARPU_R }
    };

    int main(int argc, char **argv) {
        
        // initialize MPI
        int thread_support;
        MPI_Init_thread(
            &argc, (char ***)&argv, MPI_THREAD_MULTIPLE, &thread_support);
        
        // initialize StarPU
        if (starpu_init(NULL) != 0)
            printf("Failed to initialize Starpu.\n");

        // initialize StarPU-MPI
        if (starpu_mpi_init(&argc, &argv, 0) != 0)
            printf("Failed to initialize Starpu-MPI.\n");
        
        // query world communicator's size
        int world_size = starpu_mpi_world_size();
        
        // initialize all data handles
        starpu_data_handle_t handles[world_size];
        for (int i = 0; i < world_size; i++) {
            
            // register a data handle that is going to be initialized later
            starpu_variable_data_register(
                &handles[i], -1, (uintptr_t) NULL, sizeof(int));
            
            // register data handle's owner and tag
            starpu_mpi_data_register(handles[i], i, i);
        }
        
        // insert tasks that initialize the data handles
        for (int i = 0; i < world_size; i++)
            starpu_mpi_task_insert(
                MPI_COMM_WORLD, &write_number_cl, 
                STARPU_EXECUTE_ON_DATA, handles[i], // data handles owner executes
                STARPU_W, handles[i], 0);
            
        // insert tasks that print the data handles
        for (int i = 0; i < world_size; i++)
            starpu_mpi_task_insert(
                MPI_COMM_WORLD, &read_number_cl, 
                STARPU_EXECUTE_ON_NODE, i,          // rank i executes
                STARPU_R, handles[(i+1)%world_size], 0);

        // unregister all data handles
        for (int i = 0; i < world_size; i++)
            starpu_data_unregister(handles[i]);
        
        // de-initialize everything
        starpu_mpi_shutdown();
        starpu_shutdown();
        MPI_Finalize();
        
        return 0;
    }

We are going to launch four MPI processes and allocate two CPU cores for each process:
    
.. code-block:: bash
    :emphasize-lines: 3-10

    $ gcc -o my_program my_program.c -lstarpu-1.3 -lstarpumpi-1.3 -lmpi -Wall
    $ STARPU_WORKERS_NOBIND=1 mpirun -n 4 --map-by :PE=2 ./my_program 
    Rank 1 writes value 1.
    Rank 0 writes value 0.
    Rank 2 writes value 2.
    Rank 3 reads value 0.
    Rank 3 writes value 3.
    Rank 0 reads value 1.
    Rank 1 reads value 2.
    Rank 2 reads value 3.

    
Accelerators
^^^^^^^^^^^^

As you may remember, a StarPU codelet included a field for CUDA implementations:

.. code-block:: c
    :linenos:
    :emphasize-lines: 11

    struct starpu_codelet
    {
        uint32_t where;
        int (*can_execute)(unsigned workerid, struct starpu_task *task, unsigned nimpl);
        enum starpu_codelet_type type;
        int max_parallelism;
        starpu_cpu_func_t cpu_func STARPU_DEPRECATED;
        starpu_cuda_func_t cuda_func STARPU_DEPRECATED;
        starpu_opencl_func_t opencl_func STARPU_DEPRECATED;
        starpu_cpu_func_t cpu_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_cuda_func_t cuda_funcs[STARPU_MAXIMPLEMENTATIONS];
        char cuda_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];
        char opencl_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_mic_func_t mic_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_mpi_ms_func_t mpi_ms_funcs[STARPU_MAXIMPLEMENTATIONS];
        const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];
        int nbuffers;
        enum starpu_data_access_mode modes[STARPU_NMAXBUFS];
        enum starpu_data_access_mode *dyn_modes;
        unsigned specific_nodes;
        int nodes[STARPU_NMAXBUFS];
        int *dyn_nodes;
        struct starpu_perfmodel *model;
        struct starpu_perfmodel *energy_model;
        unsigned long per_worker_stats[STARPU_NMAXWORKERS];
        const char *name;
        unsigned color;
        int flags;
        int checked;
    };

Unfortunately we do not have time to cover all the complexities that come with offloading computations to GPUs.
Instead, we will simply consider the following example:

.. code-block:: c
    :linenos:
    :emphasize-lines: 9-12,14-21,25

    #include <stdio.h>
    #include <starpu.h>

    void hello_world_cpu(void *buffers[], void *cl_arg)
    {
        printf("The host says, Hello world!\n");
    }

    __global__ void say_hello()
    {
        printf("A device says, Hello world!\n");
    }

    void hello_world_cuda(void *buffers[], void *cl_arg)
    {
        cudaStream_t stream = starpu_cuda_get_local_stream();
        say_hello<<<1, 1 , 0, stream>>>();
        cudaError err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            STARPU_CUDA_REPORT_ERROR(err);
    }

    struct starpu_codelet hello_world_cl = {
        .cpu_funcs = { hello_world_cpu },
        .cuda_funcs = { hello_world_cuda }
    };

    int main()
    {
        if (starpu_init(NULL) != 0)
            printf("Failed to initialize Starpu.\n");

        starpu_task_insert(&hello_world_cl, 0);

        starpu_task_wait_for_all();
        starpu_shutdown();

        return 0;
    }

That is, we must a task implementation (:code:`hello_world_cuda`) that inserts the CUDA kernel to the provided local CUDA stream.
Note how the naive :code:`eager` scheduler prefers to use the CPU implementation where as the GPU-aware :code:`dm` scheduler prefers the GPU:
    
.. code-block:: bash
    :emphasize-lines: 3,5

    $ nvcc -o my_program my_program.cu -lstarpu-1.3 -Xcompiler="-Wall"
    $ STARPU_SCHED=eager ./my_program 
    The host says, Hello world!
    $ STARPU_SCHED=dm ./my_program 
    A device says, Hello world!

If we want to obtain a reasonable performance using GPUs, we must define a performance model for each codelet:

.. code-block:: c
    :linenos:
    :emphasize-lines: 24

    struct starpu_codelet
    {
        uint32_t where;
        int (*can_execute)(unsigned workerid, struct starpu_task *task, unsigned nimpl);
        enum starpu_codelet_type type;
        int max_parallelism;
        starpu_cpu_func_t cpu_func STARPU_DEPRECATED;
        starpu_cuda_func_t cuda_func STARPU_DEPRECATED;
        starpu_opencl_func_t opencl_func STARPU_DEPRECATED;
        starpu_cpu_func_t cpu_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_cuda_func_t cuda_funcs[STARPU_MAXIMPLEMENTATIONS];
        char cuda_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_opencl_func_t opencl_funcs[STARPU_MAXIMPLEMENTATIONS];
        char opencl_flags[STARPU_MAXIMPLEMENTATIONS];
        starpu_mic_func_t mic_funcs[STARPU_MAXIMPLEMENTATIONS];
        starpu_mpi_ms_func_t mpi_ms_funcs[STARPU_MAXIMPLEMENTATIONS];
        const char *cpu_funcs_name[STARPU_MAXIMPLEMENTATIONS];
        int nbuffers;
        enum starpu_data_access_mode modes[STARPU_NMAXBUFS];
        enum starpu_data_access_mode *dyn_modes;
        unsigned specific_nodes;
        int nodes[STARPU_NMAXBUFS];
        int *dyn_nodes;
        struct starpu_perfmodel *model;
        struct starpu_perfmodel *energy_model;
        unsigned long per_worker_stats[STARPU_NMAXWORKERS];
        const char *name;
        unsigned color;
        int flags;
        int checked;
    };
    
In the simplest case, the model can use the run time history to predict the execution times:

.. code-block:: c
    :linenos:
    
    struct starpu_perfmodel model = {
        .type = STARPU_HISTORY_BASED
    };
    
StarPU also supports regression based performance models (:code:`STARPU_REGRESSION_BASED`, :code:`STARPU_NL_REGRESSION_BASED`, :code:`STARPU_MULTIPLE_REGRESSION_BASED`):

.. code-block:: c
    :linenos:
    
    struct starpu_perfmodel model = {
        .type = STARPU_REGRESSION_BASED
    };
    
By default, StarPU calculates the model argument from the amount of memory required to store all involved data handles.
However, a programmer may provide a custom function for this.
