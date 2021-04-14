StarPU
------

.. objectives::

 - Understand the basics of StarPU
 - Understand how StarPU support distributed memory
 - Understand how StarPU support GPUs

`StarPU <https://starpu.gitlabpages.inria.fr/>`__ (*A Unified Runtime System for Heterogeneous Multicore Architectures*) is a programming API for shared-memory and distributed-memory parallel programming in C and C++ languages.
StarPU can also be used through OpenMP pragmas and provides the necessary routines and support to natively access most of its functionalities from Fortran 2008+ codes.
StarPU supports accelerator devices such as GPUs.

Benefits and downsides
^^^^^^^^^^^^^^^^^^^^^^

StarPU has several benefits:

 1. The StarPU API is very **rich** and partly **modular**.
 
    - A programmer can, for example, implement a custom modular scheduler.
    
 2. StarPU uses **performance models** to predict which computational resource is optimal in a given situation.
 
    - A programmer does not need to decide between a GPU and a GPU. 
    - StarPU does scheduling decisions dynamically during run time.
    
 3. StarPU supports distributed memory.
 
    - The MPI communications are done **implicitly**.
    - A programmer only needs to specify the data distribution.
    - The communication pattern is automatically derived from the task graph and the data distribution.

However, when compared to OpenMP tasks, StarPU has a few downsides:

 1. The API is more complex and requires more involvement from the programmer.
 
 2. The performance models are only as good as their calibration data.
 
 3. OpenMP is a well-established standard. StarPU is not.

How to use StarPU
^^^^^^^^^^^^^^^^^

Three ways to use StarPU:

 1. C extension: Simple pragma-based approach (does not work with all compilers).
 2. StarPU API: More powerful but complex (“traditional” C library).
 3. StarPU API with helper functions: Simplifies certain things (**our choice**).

Initialization and shutdown
"""""""""""""""""""""""""""
 
Since StarPU is a C library, **a programmer must initialize and shutdown the runtime system**.
The following three basic steps are necessary:

 1. Include `starpu.h` header file:
 
    .. code-block:: c
    
        #include <starpu.h>
 
 2. Initialize StarPU runtime system by calling `starpu_init`:
 
    .. code-block:: c
 
        int starpu_init (struct starpu_conf *conf)

 3. Finally, shutdown StarPU runtime sytem by calling `starpu_shutdown()`:
 
    .. code-block:: c
 
        void starpu_shutdown (void)

    - Either unregister all data handles using blocking calls (to be covered) or call `starpu_task_wait_for_all()` before shutting StarPU down:
     
       .. code-block:: c

           int starpu_task_wait_for_all (void)

Hello world
"""""""""""

Lets consider the following "Hello world" program:

.. code-block:: c
    :linenos:
    :emphasize-lines: 2,4-7,9-11,15-16,18,20-21

    #include <stdio.h>
    #include <starpu.h>

    void hello_world_cpu(void *buffers[], void *cl_arg)
    {
        printf("Hello world!\n");
    }

    struct starpu_codelet hello_world_cl = {
        .cpu_funcs = { hello_world_cpu }
    };

    int main()
    {
        if(starpu_init(NULL) != 0)
            printf("Failed to initialize Starpu.\n");

        starpu_task_insert(&hello_world_cl, 0);

        starpu_task_wait_for_all();
        starpu_shutdown();

        return 0;
    }

Clearly this example is much more complicated that the corresponding OpenMP "Hello world" program:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4,6

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel
        {
            #pragma omp task
            printf("Hello world!\n");
        }
        return 0;
    }

In addition to initialising and shutting down StarPU, we have also introduced a separate `hello_world_cpu` function than contains the `printf` statement and a `hello_world_cl` C struct that contains a pointer to the `hello_world_cpu` function.
The task itself is created using the `starpu_task_insert` function.

For compilation, we must link the binary with the StarPU library:
    
.. code-block:: bash
    :emphasize-lines: 7
    
    $ gcc -o starpu_program starpu_program.c -Wall -lstarpu-1.3
    $ ./starpu_program
    [starpu][initialize_lws_policy] Warning: you are running the default lws scheduler, 
    which is not a very smart scheduler, while the system has GPUs or several memory 
    nodes. Make sure to read the StarPU documentation about adding performance models 
    in order to be able to use the dmda or dmdas scheduler instead.
    Hello world!

The printed warning is related to the fact that StarPU's default scheduler is not smart enough to handle GPUs correctly.

.. challenge::

    Modify the StarPU "Hello world" program such that 8 tasks are created.

.. solution::

    The simplest solution is to introduce a `for` loop:

    .. code-block:: c
        :linenos:
        :emphasize-lines: 18

        #include <stdio.h>
        #include <starpu.h>

        void hello_world_cpu(void *buffers[], void *cl_arg)
        {
            printf("Hello world!\n");
        }

        struct starpu_codelet hello_world_cl = {
            .cpu_funcs = { hello_world_cpu }
        };

        int main()
        {
            if(starpu_init(NULL) != 0)
                printf("Failed to initialize Starpu.\n");

            for (int i = 0; i < 8; i++)
                starpu_task_insert(&hello_world_cl, 0);

            starpu_task_wait_for_all();
            starpu_shutdown();

            return 0;
        }
    
    .. code-block:: bash
        :emphasize-lines: 3-10

        $ gcc -o starpu_program starpu_program.c -Wall -lstarpu-1.3
        $ ./starpu_program
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!

Codelets and tasks
^^^^^^^^^^^^^^^^^^

When StarPU is initialized, the creates a set of **worker threads**.
Usually each CPU core gets its own worker thread.
Depending on the configuration, one or more CPU cores (and GPU worker threads) are allocated for managing any GPUs.
All tasks are placed into a task pool from which the worker threads pick tasks as they become ready for scheduling.

Each **task type** is defined within a StarPU **codelet**:

.. code-block:: c
    :linenos:
    :emphasize-lines: 10,11,13

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

Each task type can have **multiple implementations**.
In the earlier "Hello world" example, the `hello_world_cl` had just one CPU implementation:

.. code-block:: c
    :linenos:
    :emphasize-lines: 1-4,7

    void hello_world_cpu(void *buffers[], void *cl_arg)
    {
        printf("Hello world!\n");
    }

    struct starpu_codelet hello_world_cl = {
        .cpu_funcs = { hello_world_cpu }
    };

In addition to having multiple CPU implementations, a codelet can contain several **CUDA implementation** (`cuda_funcs`) and **OpenCL implementation** (`opencl_funcs`).
All functions that implement the codelet have a similar prototype:

.. code-block:: c

    typedef void (*starpu_cpu_func_t)(void **, void*);
    typedef void (*starpu_cuda_func_t)(void **, void*);
    typedef void (*starpu_opencl_func_t)(void **, void*);

.. challenge::

    Modify the "Hello world" program as follows:
    
     1. Create a second implementation called `hi_world_cpu` that prints "Hi!".
     2. Add the new implementation to the codelet as a first implementation.
    
    **Hint:** The `cpu_funcs` field is a regular C array.

.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 9-12,15

        #include <stdio.h>
        #include <starpu.h>

        void hello_world_cpu(void *buffers[], void *cl_arg)
        {
            printf("Hello world!\n");
        }

        void hi_world_cpu(void *buffers[], void *cl_arg)
        {
            printf("Hi!\n");
        }

        struct starpu_codelet hello_world_cl = {
            .cpu_funcs = { hi_world_cpu, hello_world_cpu }
        };

        int main()
        {
            if(starpu_init(NULL) != 0)
                printf("Failed to initialize Starpu.\n");

            starpu_task_insert(&hello_world_cl, 0);

            starpu_task_wait_for_all();
            starpu_shutdown();

            return 0;
        }

    
    .. code-block:: bash
        :emphasize-lines: 3

        $ gcc -o starpu_program starpu_program.c -Wall -lstarpu-1.3
        $ ./starpu_program
        Hi!

Data handles and arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed memory
^^^^^^^^^^^^^^^^^^

Accelerators
^^^^^^^^^^^^

