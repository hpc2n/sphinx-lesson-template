OpenMP task basics (part 2)
---------------------------

.. objectives::

 - Learn about :code:`taskloop` and :code:`taskgroup` constructs.
 - Learn about :code:`depend`, :code:`priority`, :code:`untied`, :code:`mergeable`, and :code:`final` clauses.

Task loop
^^^^^^^^^

If is very commont that a set of tasks is created inside a loop:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5-6
    
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < n; i++) {
            #pragma omp task
            task_implementation(data[i]);
        }
    }
    
This is not always very convenient.
For example, the resulting task granularity can be too fine and we may want to combine several loop iterations into a single tasks:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4-5,8-9
    
    #pragma omp parallel
    #pragma omp single
    {
        int per_task = 10;
        for (int i = 0; i < n; i += per_task) {
            #pragma omp task
            {
                for (int j = 0; j < per_task && i+j < n; j++)
                    task_implementation(data[i+j]);
            }
        }
    }

As you can clearly see, the resulting code is quite complicated and easy to implement incorrectly. 
Fortunately the :code:`taskloop` construct can be used to solve this issue:
    
.. code-block:: c

    #pragma omp taskloop [clause[[,] clause] ...] new-line 
        for-loops

The construct specifies that **the iterations of the loops will be executed in parallel using tasks**.
We can thus write the earlier example in a much shorter form:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4
    
    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp taskloop
        for (int i = 0; i < n; i++)
            task_implementation(data[i]);
    }

Unless otherwise specified, the number of iterations assigned to each task is decided by the OpenMP implementation.
We can change this behaviour with clauses:
        
.. code-block:: c
    :emphasize-lines: 9,10

    if([ taskloop :] scalar-expression) 
    shared(list) 
    private(list) 
    firstprivate(list) 
    lastprivate(list) 
    reduction([default ,]reduction-identifier : list) 
    in_reduction(reduction-identifier : list) 
    default(shared | none) 
    grainsize(grain-size) 
    num_tasks(num-tasks) 
    collapse(n) 
    final(scalar-expr) 
    priority(priority-value) 
    untied 
    mergeable 
    nogroup 
    allocate([allocator :] list)

In particular, the :code:`grainsize` sets the number of iterations assigned to each tasks and the :code:`num_tasks` sets the number of tasks generated.
    
Task group
^^^^^^^^^^

The :code:`taskwait` construct specifies that the current task region is suspended until the completion of child tasks of the current task. 
However, the construct **does not** specify that the current task region is suspended until the completion of **descendants of the child tasks**:

.. code-block:: c
    :linenos:
    :emphasize-lines: 9-10,15,17

    #include <stdio.h>

    int main() {
        #pragma omp parallel
        #pragma omp single
        {
            #pragma omp task
            {
                #pragma omp task
                printf("Hello.\n");

                printf("Hi.\n");
            }

            #pragma omp taskwait

            printf("Goodbye.\n");
        }

        return 0;
    }

In the above example, the tasks that prints the :code:`Hello` line is a descendant of the child task that prints the :code:`Hi` line.
As we can see, it is possible that the descendant gets executed after the :code:`taskwait` region:
    
.. code-block:: bash
    :emphasize-lines: 4-5
    
    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Hi.
    Goodbye.
    Hello.

Tasks and their descendent tasks can be synchronized by containing them in a :code:`taskgroup` region.
The :code:`taskgroup` construct specifies a wait on completion of** child tasks** of the current task and **their descendent tasks**:

.. code-block:: c

    #pragma omp taskgroup [clause[[,] clause] ...] new-line 
        structured-block
        
Note that the :code:`taskgroup` construct is **not** a standalone construct.
Instead, we must enclose the task generating region with it.
All tasks generated inside a :code:`taskgroup` region are waited for at the end of the region.

.. challenge::

    Modify the following code such that the two tasks are enclosed inside a :code:`taskgroup` region:
    
    .. code-block:: c
        :linenos:

        #include <stdio.h>

        int main() {
            #pragma omp parallel
            #pragma omp single
            {
                #pragma omp task
                {
                    #pragma omp task
                    printf("Hello.\n");

                    printf("Hi.\n");
                }

                printf("Goodbye.\n");
            }

            return 0;
        }
        
.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 8-9,17

        #include <stdio.h>

        int main() {
            #pragma omp parallel
            #pragma omp single nowait
            {

                #pragma omp taskgroup
                {
                    #pragma omp task
                    {
                        #pragma omp task
                        printf("Hello.\n");

                        printf("Hi.\n");
                    }
                }

                printf("Goodbye.\n");
            }

            return 0;
        }

    .. code-block:: bash
        :emphasize-lines: 3-5,7-9
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        Hi.
        Hello.
        Goodbye.
        $ ./my_program 
        Hello.
        Hi.
        Goodbye.

Depend clause
^^^^^^^^^^^^^

Up to this point, we have only discussed tasks that are either mutually independent or are related to each other due to the fact that they are generated in a nested manner.
In the earlier lecture, we talked about **task dependencies**. 
Since OpenMP 4.5, most task-related OpenMP constructs have accepted the :code:`depend` clause:

.. code-block:: c

    depend([depend-modifier,]dependence-type : locator-list)
    
where :code:`dependence-type` is one of the following: 

.. code-block:: c
    :emphasize-lines: 1-3

    in 
    out 
    inout 
    mutexinoutset 
    depobj
    
The most relevant ones of these are the following:

:in:        Input variable(s).
:out:       Output variable(s).
:inout:     Input and output variable(s).

The :code:`locator-list` argument lists all involved variables: :code:`var1, var2, ..., varN`.
A construct can have **multiple** :code:`depend` clauses, one for each :code:`dependence-type`.
The :code:`depend` clause is much more powerful than this but during this course we are going to use only the basic functionality.

As an example, consider the following ill-defined program:

.. code-block:: c
    :linenos:
    :emphasize-lines: 9-10,12-16,18-19

    #include <stdio.h>

    int main() {
        int number;

        #pragma omp parallel
        #pragma omp single nowait
        {
            #pragma omp task
            number = 1;

            #pragma omp task
            {
                printf("I think the number is %d\n", number);
                number++;
            }

            #pragma omp task
            printf("I think the final number is %d\n", number);
        }

        return 0;
    }

As expected, the result is not well-defined:
    
.. code-block:: bash
    :emphasize-lines: 3-4, 6-7
    
    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    I think the number is 1
    I think the final number is 1
    $ ./my_program 
    I think the final number is 1
    I think the number is 1

We can fix the issue by defining input and output variables for each task:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 9,12,18

    #include <stdio.h>

    int main() {
        int number;

        #pragma omp parallel
        #pragma omp single nowait
        {
            #pragma omp task depend(out: number)
            number = 1;

            #pragma omp task depend(inout: number)
            {
                printf("I think the number is %d\n", number);
                number++;
            }

            #pragma omp task depend(in: number)
            printf("I think the final number is %d\n", number);
        }

        return 0;
    }

That is, 

 - the first task is going to write into the variable :code:`number`,
 - the second task is going to read and write from/into the variable :code:`number`, and
 - the third task is going to read from the variable :code:`number`.

These clauses force the OpenMP implementation to execute the tasks in an order that respects the induced task dependencies:
 
.. code-block:: bash
    :emphasize-lines: 3-4

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    I think the number is 1
    I think the final number is 2

.. challenge::

    Parallelize the following program using tasks: 

    .. code-block:: c
        :linenos:

        #include <stdio.h>

        #define N 15

        int main() {
            int fib_numbers[N];

            fib_numbers[0] = 1;
            fib_numbers[1] = 1;

            for (int i = 2; i < N; i++) {
                fib_numbers[i] = fib_numbers[i-1] + fib_numbers[i-2];
            }

            printf("The Fibonacci numbers are:");
            for (int i = 0; i < N; i++)
                printf(" %d", fib_numbers[i]);
            printf("\n");

            return 0;
        }
        
    .. code-block:: bash
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        The Fibonacci numbers are: 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610
        
    **Hint:** :code:`locator-list` can contain array elements.
    
.. solution::

    .. code-block:: c
        :linenos:
    
        #include <stdio.h>

        #define N 15

        int main() {
            int fib_numbers[N];

            #pragma omp parallel
            #pragma omp single
            {

                #pragma omp task default(none) shared(fib_numbers) \
                    depend(out: fib_numbers[0])
                fib_numbers[0] = 1;
            
                #pragma omp task default(none) shared(fib_numbers) \
                    depend(out: fib_numbers[1])
                fib_numbers[1] = 1;

                for (int i = 2; i < N; i++) {
                    #pragma omp task \
                        default(none) shared(fib_numbers) firstprivate(i) \
                        depend(in: fib_numbers[i-1], fib_numbers[i-2]) \
                        depend(out: fib_numbers[i])
                    fib_numbers[i] = fib_numbers[i-1] + fib_numbers[i-2];
                }
            }

            printf("The Fibonacci numbers are:");
            for (int i = 0; i < N; i++)
                printf(" %d", fib_numbers[i]);
            printf("\n");

            return 0;
        }
        
    .. code-block:: bash
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        The Fibonacci numbers are: 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610

Priority clause
^^^^^^^^^^^^^^^

As discussed during an earlier lecture, we can give each task a **priority** in an attempt to help the runtime system to schedule the tasks in a more optimal order.

In OpenMP, the priority is given using the :code:`priority(priority-value)` clause.
The :code:`priority-value` is a **non-negative integer expression** and higher value implies higher priority.

Untied clause and taskyield construct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Threads are allowed to **suspend** the current task region at a **task scheduling point** in order to execute a different task.
A task scheduling point can occur

 - during the generation of an explicit task,
 - the point immediately following the generation of an explicit task,
 - after the point of completion of the structured block associated with a task,
 - in a :code:`taskyield` region,
 - in a :code:`taskwait` region,
 - at the end of a taskgroup region,
 - in an implicit barrier region, and
 - in an explicit barrier region.

In particular, we can use the :code:`taskyield` construct to force a task scheduling point.

.. code-block:: c

    #pragma omp taskyield new-line
 
Note that the above list is not complete.

By default, task regions are **tied** to the the initially assigned thread and **only** the initially assigned thread can later resume the execution of the suspended task region.
This behaviour can be changed with the :code:`untied` in which case any thread is allowed to resume the execution.

Mergeable and final clauses
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`mergeable` and :code:`final` clauses can be used to reduce the number of generated tasks in deep nested task generation trees.

First, we need more terminology:

:Undeferred task:   A task for which execution is not deferred with respect to its generating task region. 
                    That is, its **generating task region is suspended until execution of the structured block associated with the undeferred task is completed**.

:Included task:     A task for which execution is sequentially **included in the generating task region**. 
                    That is, an included task is **undeferred** and executed by the encountering thread.
                    
:Merged task:       A task for which the data environment, inclusive of ICVs, is the same as that of its generating task region.

:Mergeable task:    A task that may be a merged task if it is an undeferred task or an included task.

:Final task:        A task that forces all of its child tasks to become final and included tasks.

The :code:`mergeable` clause indicates that the task is mergeable.

If :code:`scalar-expression` is evaluated as true, then the :code:`final(scalar-expression)` clause indicates that the task is final.
