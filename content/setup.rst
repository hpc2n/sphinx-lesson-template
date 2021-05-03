Introduction to Kebnekaise
--------------------------

.. objectives::

 - Learn how to load the necessary modules on Kebnekaise.
 - Learn how to compile C code on Kebnekaise.
 - Learn how to compile CUDA code on Kebnekaise.
 - Learn how to place jobs to the batch queue.
 - Learn how to use the course project and reservations.

Modules and toolchains
^^^^^^^^^^^^^^^^^^^^^^

You need to **load the correct toolchain** before compiling your code on Kebnekaise.

The **available modules** are listed using the :code:`ml avail` command:

.. code-block:: bash

    $ ml avail
    ------------------------- /hpc2n/eb/modules/all/Core --------------------------
    Bison/3.0.5                        fosscuda/2020a
    Bison/3.3.2                        fosscuda/2020b        (D)
    Bison/3.5.3                        gaussian/16.C.01-AVX2
    Bison/3.7.1                (D)     gcccuda/2019b
    CUDA/8.0.61                        gcccuda/2020a
    CUDA/10.1.243              (D)     gcccuda/2020b         (D)
    ...

The list shows the modules you can load directly, and so may change if you have loaded modules.

In order to see all the modules, including those that have prerequisites to load, use the command :code:`ml spider`. Many types of application software fall in this category. 

You can find more **information** regarding a particular module using the :code:`ml spider <module>` command:

.. code-block:: bash

    $ ml spider MATLAB

    ---------------------------------------------------------------------------
    MATLAB: MATLAB/2019b.Update2
    ---------------------------------------------------------------------------
        Description:
        MATLAB is a high-level language and interactive environment that
        enables you to perform computationally intensive tasks faster than
        with traditional programming languages such as C, C++, and Fortran.


        This module can be loaded directly: module load MATLAB/2019b.Update2

        Help:
        Description
        ===========
        MATLAB is a high-level language and interactive environment
        that enables you to perform computationally intensive tasks faster than with
        traditional programming languages such as C, C++, and Fortran.
        
        
        More information
        ================
        - Homepage: http://www.mathworks.com/products/matlab

You can **load** the module using the :code:`ml <module>` command:

.. code-block:: bash

    $ ml foss MATLAB/2019b.Update2

You can **list loaded modules** using the :code:`ml` command:

.. code-block:: bash

    $ ml

    Currently Loaded Modules:
     1) snicenvironment     (S)   7) libevent/2.1.11    13) PMIx/3.0.2
     2) systemdefault       (S)   8) numactl/2.0.12     14) impi/2018.4.274
     3) GCCcore/8.2.0             9) XZ/5.2.4           15) imkl/2019.1.144
     4) zlib/1.2.11              10) libxml2/2.9.8      16) intel/2019a
     5) binutils/2.31.1          11) libpciaccess/0.14  17) MATLAB/2019b.Update2
     6) iccifort/2019.1.144      12) hwloc/1.11.11

    Where:
     S:  Module is Sticky, requires --force to unload or purge
    
You can **unload all modules** using the :code:`ml purge` command:

.. code-block:: bash

    $ ml purge
    The following modules were not unloaded:
      (Use "module --force purge" to unload all):

      1) systemdefault   2) snicenvironment

Note that the :code:`ml purge` command will warn that two modules were not unloaded. 
This is normal and you should **NOT** force unload them.

.. challenge::

    1. Load the FOSS CUDA toolchain for source code compilation:
 
       .. code-block:: bash
       
            $ ml purge
            $ ml fosscuda/2020b buildenv
    
       The :code:`fosscuda` module loads the GNU compiler, the CUDA SDK and several other libraries. 
       The :code:`buildenv` module sets certain environment variables that are necessary for source code compilation.
       
    2. Investigate which modules were loaded.
       
    3. Purge all modules.
       
    4. Find the latest FOSS toolchain (:code:`foss`). Load it and the :code:`buildenv` module. 
       Investigate the loaded modules.
       Purge all modules.

Compile C code
^^^^^^^^^^^^^^

Once the correct toolchain (:code:`foss`) has been loaded, when can compile C source files (:code:`*.c`) with the GNU compiler:

.. code-block:: bash

    $ gcc -o <binary name> <sources> -Wall

The :code:`-Wall` causes the compiler to print additional warnings.

.. challenge::

    Compile the following "Hello world" program:
    
    .. code-block:: c
        :linenos:
    
        #include <stdio.h>

        int main() {
            printf("Hello world!\n");
            return 0;
        }

Compile CUDA code
^^^^^^^^^^^^^^^^^

Once the correct toolchain (:code:`fosscuda`) has been loaded, when can compile CU source files (:code:`*.cu`) with the :code:`nvcc` compiler:

.. code-block:: bash

    $ nvcc -o <binary name> <sources> -Xcompiler="-Wall"

This passes the :code:`-Wall` flag to :code:`g++`. The flag causes the compiler to print extra warnings.
    
.. challenge::

    Compile the following "Hello world" program:
    
    .. code-block:: c
        :linenos:
    
        #include <stdio.h>

        __global__ void say_hello()
        {
            printf("A device says, Hello world!\n");
        }

        int main()
        {
            printf("The host says, Hello world!\n");
            say_hello<<<1,1>>>();
            cudaDeviceSynchronize();
            return 0;
        }

Course project and reservation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
During the course, you can use the **course reservations** (snic2021-22-272-cpu-day[1|2|3] and snic2021-22-272-gpu-day[1|2|3]) to get faster access to the compute nodes. 
The reservations are valid during the time 9:00-13:00 on each of the three days (10-12 May 2021). 
Note that capitalization matters for reservations!

+-----------+--------------------------+--------------------------+
| Day       | CPU only                 | CPU + GPU                |
+===========+==========================+==========================+
| Monday    | snic2021-22-272-cpu-day1 | snic2021-22-272-gpu-day1 |
+-----------+--------------------------+--------------------------+
| Tuesday   | snic2021-22-272-cpu-day2 | snic2021-22-272-gpu-day1 |
+-----------+--------------------------+--------------------------+
| Wednesday | snic2021-22-272-cpu-day3 | snic2021-22-272-gpu-day1 |
+-----------+--------------------------+--------------------------+

Note that jobs that are submitted using a reservation are not scheduled outside the reservation time window. 
You can, however, submit jobs without the reservation as long as you are a member of an active project. 
The **course project** :code:`SNIC2021-22-272` is valid until 2021-06-01.

Submitting jobs
^^^^^^^^^^^^^^^

The jobs are **submitted** using the :code:`srun` command:

.. code-block:: bash

    $ srun --account=<account> --ntasks=<task count> --time=<time> <command>

This places the command into the batch queue.
The three arguments are the project number, the number of tasks, and the requested time allocation.
For example, the following command prints the uptime of the allocated compute node:

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --ntasks=1 --time=00:00:15 uptime
    srun: job 12727702 queued and waiting for resources
    srun: job 12727702 has been allocated resources
     11:53:43 up 5 days,  1:23,  0 users,  load average: 23,11, 23,20, 23,27

Note that we are using the course project, the number of tasks is set to one, and we are requesting 15 seconds.

When the **reservation** is valid, you can specify it using the :code:`--reservation=<reservation>` argument:

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-cpu-day1 --ntasks=1 --time=00:00:15 uptime
     11:58:43 up 6 days,  1:23,  0 users,  load average: 23,11, 22,20, 21,27

were N in dayN is either 1, 2, 3 and cpu can be replaced with gpu if you are running a GPU job. 

We could submit **multiple tasks** using the :code:`--ntasks=<task count>` argument:

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-cpu-day1 --ntasks=4 --time=00:00:15 uname -n
    b-cn0932.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    
Note that all task are running on the same node.
We could request **multiple CPU cores** for each task using the :code:`--cpus-per-task=<cpu count>` argument:

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-cpu-day1 --ntasks=4 --cpus-per-task=14 --time=00:00:15 uname -n
    b-cn0935.hpc2n.umu.se
    b-cn0935.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se

If you want to measure the performance, it is advisable to request an **exclude access** to the compute nodes (:code:`--exclude`):

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-cpu-day1 --ntasks=4 --cpus-per-task=14 --exclude --time=00:00:15 uname -n
    b-cn0935.hpc2n.umu.se
    b-cn0935.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    b-cn0932.hpc2n.umu.se
    
Finally, we could request a **single Nvidia Tesla V100 GPU** and 14 CPU cores using the :code:`--gres=gpu:v100:1,gpuexcl` argument:

.. code-block:: bash

    $ srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-gpu-day1 --ntasks=1 --gres=gpu:v100:1,gpuexcl --time=00:00:15 nvidia-smi
    Wed Apr 21 12:59:15 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.67       Driver Version: 460.67       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-PCIE...  On   | 00000000:58:00.0 Off |                    0 |
    | N/A   33C    P0    26W / 250W |      0MiB / 16160MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

    
.. challenge::

    Run both "Hello world" programs on the the compute nodes.
 
Aliases
^^^^^^^

In order to save time, you can create an **alias** for a command:

.. code-block:: bash

    $ alias <alist>="<command>"

For example:

.. code-block:: bash

    $ alias run_full="srun --account=SNIC2021-22-272 --reservation=snic2021-22-272-cpu-day1 --ntasks=1 --cpus-per-task=28 --time=00:05:00"
    $ run_full uname -n
    b-cn0932.hpc2n.umu.se

Batch files
^^^^^^^^^^^

I is often more convenient to write the commands into a **batch file**.
For example, we could write the following to a file called :code:`batch.sh`:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH --account=SNIC2021-22-272
    #SBATCH --reservation=snic2021-22-272-cpu-day1
    #SBATCH --ntasks=1
    #SBATCH --time=00:00:15

    ml purge
    ml foss/2020b

    uname -n

Note that the same arguments that were earlier passed to the :code:`srun` command are now given as comments.
It is highly advisable to purge all loaded modules and re-load the required modules as the job inherits the environment.
The batch file is submitted using the :code:`sbatch <batch file>` command:
    
.. code-block:: bash

    sbatch batch.sh 
    Submitted batch job 12728675

By default, the output is directed to the file :code:`slurm-<job_id>.out`, where :code:`<job_id>` is the **job id** returned by the :code:`sbatch` command:

.. code-block:: bash

    $ cat slurm-12728675.out 
    The following modules were not unloaded:
     (Use "module --force purge" to unload all):

     1) systemdefault   2) snicenvironment
    b-cn0102.hpc2n.umu.se
    
.. challenge::
        
    Write two batch files that run both "Hello world" programs on the the compute nodes.
        
Job queue
^^^^^^^^^
        
You can **investigate the job queue** with the :code:`squeue` command:

.. code-block:: bash

    $ squeue -u $USER

If you want an estimate for when the job will start running, you can give the :code:`squeue` command the argument :code:`--start`. 

You can **cancel** a job with the :code:`scancel` command:

.. code-block:: bash

    $ scancel <job_id>
