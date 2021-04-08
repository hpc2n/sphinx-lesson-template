Setup
-----

.. objectives::

 - Learn how to load the necessary modules on Kebnekaise.
 - Learn how to compile CUDA code.
 - Learn how to place jobs to the batch queue.
 - Learn how to use the course reservations.

Materials
^^^^^^^^^
 
.. challenge::

    Clone course materials::

        $ git clone https://github.com/hpc2n/Task-based-parallelism.git

Modules and compiling
^^^^^^^^^^^^^^^^^^^^^
        
.. challenge::

    Load the necessary modules (only on Kebnekaise)::
 
        $ ml purge
        $ ml fosscuda/2019b buildenv
    
    The `purge` command unload existing modules. Note that the `purge` command
    will warn that two modules were not unloaded. This is normal and you should
    **NOT** force unload them. The `fosscuda` module loads the GNU compiler,
    the CUDA SDK and several other libraries. The `buildenv` module sets certain
    environment variables.
 
    Compile the `hello.cu` source file with `nvcc` compiler::

        $ nvcc -o hello hello.cu

    In some situations, it is beneficial to pass additional arguments to the
    host compiler (`g++` in this case)::

        $ nvcc -o hello hello.cu -Xcompiler="-Wall"

    This passes the `-Wall` flag to `g++`. The flag causes the compiler to print
    extra warnings.

Jobs and reservation
^^^^^^^^^^^^^^^^^^^^
    
During the course, you can use the course reservations (6 Nvidia V100 GPUs) to
get faster access to the GPUs. The reservation `snic2020-9-161-day1` is valid
during Wednesday and the reservation `snic2020-9-161-day2` is valid during
Thursday. The reservations are valid from 08:45 to 17:30. 

Note that jobs that are submitted using a reservation are not scheduled outside
the reservation time window. You can, however, submit jobs without the
reservation as long as you are a member of an active project. The project
`SNIC2020-9-161` is valid until 2020-12-01.

.. challenge::

    Run the program::
 
        $ srun --account=SNIC2020-9-161 --reservation=snic2020-9-161-day1 --ntasks=1 --gres=gpu:v100:1,gpuexcl --time=00:02:00 ./hello
        srun: job .... queued and waiting for resources
        srun: job .... has been allocated resources
        Host says, Hello world!
        GPU says, Hello world!
    
    This can take a few minutes if several people are trying to use the GPUs
    simultaneously. 
    
The `srun` command places the program into the batch queue,

    - `--account=SNIC2020-9-161` sets the account number,
    - `--reservation=snic2020-9-161-day1` sets the reservation,
    - `--ntasks=1` sets the number of tasks to one,
    - `--gres=gpu:v100:1,gpuexcl` requests exclusive access to a single Nvidia
      Tesla V100 GPU (and 14 CPU cores), 
    - `--time=00:02:00` sets the maximum run time to two minutes (please do not
      create long jobs),
    
and the last argument the is the program itself.
 
Alias
^^^^^
 
You can also create an **alias** for the command::
    
        $ alias run_gpu="srun --account=SNIC2020-9-161 --reservation=snic2020-9-161-day1 --ntasks=1 --gres=gpu:v100:1,gpuexcl --time=00:02:00"
        $ run_gpu ./hello
        Host says, Hello world!
        GPU says, Hello world!

Batch files
^^^^^^^^^^^
        
.. challenge::
        
    Create a file called `batch.sh` with the following contents::
 
        #!/bin/bash
        #SBATCH --account=SNIC2020-9-161
        #SBATCH --reservation=snic2020-9-161-day1
        #SBATCH --ntasks=1
        #SBATCH --gres=gpu:v100:1,gpuexcl
        #SBATCH --time=00:02:00

        ml purge
        ml fosscuda/2019b buildenv

        ./hello
    
    Submit the batch file::
    
        $ sbatch batch.sh 
        Submitted batch job ....

Job queue
^^^^^^^^^
        
You can investigate the job queue with the following command::

    $ squeue -u $USER

If you want an estimate for when the job will start running, you can
give the `squeue` command the argument `--start`. By default, the output of
the batch file goes to `slurm-<job id>.out`.
