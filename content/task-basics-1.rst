OpenMP task basics (part 1)
---------------------------

.. objectives::

 - Understand the basics of OpenMP tasks

Task construct
^^^^^^^^^^^^^^^

The simplest way to create an (explicit) task in OpenMP is the **task** pragma:

.. code-block:: c

    #pragma omp task [clause[ [,] clause] ... ] new-line 
        structured-block

The tread that encounters the task pragma creates an (explicit) tasks from the structured block.
The encountering thread **may execute the task immediately** or **defer its execution to one of the other threads in the team**.
If a task construct is encountered outside a parallel construct, then the structured block is executed immediately by the encountering thread.

The pragma accepts a set of clauses:

.. code-block:: c
    :emphasize-lines: 1,4,6,7,8

    if([ task :] scalar-expression) 
    final(scalar-expression) 
    untied 
    default(shared | none) 
    mergeable 
    private(list) 
    firstprivate(list) 
    shared(list) 
    in_reduction(reduction-identifier : list) 
    depend([depend-modifier,] dependence-type : locator-list) 
    priority(priority-value) 
    allocate([allocator :] list) 
    affinity([aff-modifier :] locator-list) 
    detach(event-handle)

We can already recognise some of the clauses.
For example, the **if** clause can be used to enable/disable the creation of the corresponding task, and the **default**, **private**, **firstprivate**, and **shared** clauses are used to control the data sharing rules.
It should be noted that some of these clauses behave slightly differently when compared the traditional OpenMP constructs.
However, for the purposes of this course, there is no difference.

Simple example
""""""""""""""

Let us return back to the earlier "Hello world" program:
    
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

Note that the task pragma is **inside a parallel construct**.
Each task in the team encounters the task construct, creates the corresponding tasks and either executes the task immediately or defer its execution to one of the other threads in the team.
Therefore, the number of tasks, and lines printed, are the same as the number of threads in the team:
    
.. code-block:: bash
    :emphasize-lines: 3-6

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Hello world!
    Hello world!
    ...
    Hello world!

Terminology
"""""""""""
