Introduction to OpenMP (part 2)
-------------------------------

.. objectives::

 - Understand the basics of OpenMP
 
Section construct
^^^^^^^^^^^^^^^^^

As we saw earlier, **all threads within the team** execute the **entire structured block** that follows a parallel construct.
Only a very limited number of parallel algorithms can be implemented in this way.
It is much more common that we have a set of mutually independent operations which we want to execute in parallel.

One way of accomplishing this are with the **sections** and **section** constructs:

.. code-block:: c

    #pragma omp sections [clause[ [,] clause] ... ] new-line 
    { 
        [#pragma omp section new-line] 
            structured-block 
        [#pragma omp section new-line 
            structured-block] 
        ...
    }

The structured blocks that follow the :code:`section` constructs inside the :code:`sections` construct are distributed among the threads within the team:

.. figure:: img/section.png
    :align: center
    :scale: 75%

Each structured block is executed only **once**:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5,7,9,11-12,14-15,17-18
    
    #include <stdio.h>

    int main() {
    
        #pragma omp parallel
        {
            printf("Everyone!\n");

            #pragma omp sections
            {
                #pragma omp section
                printf("Only me!\n");
                
                #pragma omp section
                printf("No one else!\n");
                
                #pragma omp section
                printf("Just me!\n");
            }
        }

        return 0;
    }

.. code-block:: bash
    :emphasize-lines: 3-9

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Everyone!
    Only me!
    No one else!
    Just me!
    Everyone!
    Everyone!
    ...

Note how the :code:`Everyone!` lines are printed multiple times but the other three lines are printed only once.
    
If we want, we can merge the :code:`parallel` and :code:`sections` constructs together:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 5
    
    #include <stdio.h>

    int main() {
    
        #pragma omp parallel sections
        {
            #pragma omp section
            printf("Only me!\n");
            
            #pragma omp section
            printf("No one else!\n");
            
            #pragma omp section
            printf("Just me!\n");
        }

        return 0;
    }
    
.. code-block:: bash
    :emphasize-lines: 3-5

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Just me!
    No one else!
    Only me!

.. challenge::

    Parallelize the following program using the :code:`sections` and :code:`section` constructs:
    
    .. code-block:: c
        :linenos:
    
        #include <stdio.h>

        int main() {
            int a, b, c, d;

            a = 5;
            b = 14;
            c = a + b;
            d = a + 44;
            printf("a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);

            return 0;
        }
        
    The program should print :code:`a = 5, b = 14, c = 19, d = 49`.
    Pay attention to the data dependencies.
    You may have to add more than one :code:`parallel` construct.
    
.. solution::

    The statements :code:`a = 5;` and :code:`b = 14;` can be executed in parallel and we therefore add one :code:`parallel sections` construct for them.
    The statements :code:`c = a + b;` and :code:`d = a + 44;` can be executed in parallel and we therefore add another :code:`parallel sections` construct for them.

    .. code-block:: c
        :linenos:
        :emphasize-lines: 6-7,8,10,12,13-14,15,17,19

        #include <stdio.h>

        int main() {
            int a, b, c, d;

            #pragma omp parallel sections
            {
                #pragma omp section
                a = 5;
                #pragma omp section
                b = 14;
            }
            #pragma omp parallel sections
            {
                #pragma omp section
                c = a + b;
                #pragma omp section
                d = a + 44;
            }
            printf("a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);

            return 0;
        }
        
    .. code-block:: bash
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program                                 
        a = 5, b = 14, c = 19, d = 49
    
Parallel loop construct
^^^^^^^^^^^^^^^^^^^^^^^

Most programs contain several loops and parallelizing these loops is often a natural way to add some parallelism to a program. 
The :code:`loop` construct does exactly that:

.. code-block:: c

    #pragma omp loop [clause[ [,] clause] ... ] new-line 
        for-loops
        
The construct tells OpenMP that the loop iterations are free of data dependencies and can therefore be executed in parallel.
The loop iterator is :code:`private` by default:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4,6

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel
        {
            #pragma omp loop
            for (int i = 0; i < 5; i++)
                printf("The loop iterator is %d.\n", i);
        }
    }
    
.. code-block:: bash
    :emphasize-lines: 3-7

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    The loop iterator is 1.
    The loop iterator is 4.
    The loop iterator is 0.
    The loop iterator is 2.
    The loop iterator is 3.

Like many other constructs, the :code:`loop` construct accepts several clauses:

.. code-block:: c
    :emphasize-lines: 2

    bind(binding) 
    collapse(n) 
    order(concurrent) 
    private(list) 
    lastprivate(list) 
    reduction([default ,]reduction-identifier : list)

In particular, the :code:`collapse` clause allows us to collapse :code:`n` nested loops into a single parallel loop.
Otherwise, only the iterations of the outermost loop are executed in parallel.

.. challenge::

    Collapse the two nested loops in the following program:
    
    .. code-block:: c
        :linenos:
    
        #include <stdio.h>

        int main() {
            #pragma omp parallel
            {
                #pragma omp loop
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        printf("The loop iterators are %d and %d.\n", i, j);
            }
        }
        
    .. code-block:: bash
        :emphasize-lines: 3-11
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        The loop iterators are 2 and 0.
        The loop iterators are 2 and 1.
        The loop iterators are 2 and 2.
        The loop iterators are 0 and 0.
        The loop iterators are 0 and 1.
        The loop iterators are 0 and 2.
        The loop iterators are 1 and 0.
        The loop iterators are 1 and 1.
        The loop iterators are 1 and 2.
        
    Note how the innermost loop is always executed sequentially.
    What changes?
    
.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 6
    
        #include <stdio.h>

        int main() {
            #pragma omp parallel
            {
                #pragma omp loop collapse(2)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        printf("The loop iterators are %d and %d.\n", i, j);
            }
        }

    .. code-block:: bash
        :emphasize-lines: 3-11
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        The loop iterators are 2 and 2.
        The loop iterators are 0 and 0.
        The loop iterators are 2 and 1.
        The loop iterators are 0 and 1.
        The loop iterators are 2 and 0.
        The loop iterators are 1 and 2.
        The loop iterators are 0 and 2.
        The loop iterators are 1 and 0.
        The loop iterators are 1 and 1.
        
    Note that the iterations from both loops are now executed in an arbitrary order.

If we want, we can merge the :code:`parallel` and :code:`loop` constructs together:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel loop
        for (int i = 0; i < 5; i++)
            printf("The loop iterator is %d.\n", i);
    }
    
.. code-block:: bash
    :emphasize-lines: 3-7

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    The loop iterator is 4.
    The loop iterator is 0.
    The loop iterator is 2.
    The loop iterator is 3.
    The loop iterator is 1.

Or use an older :code:`for` construct:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel for
        for (int i = 0; i < 5; i++)
            printf("The loop iterator is %d.\n", i);
    }
    
.. code-block:: bash
    :emphasize-lines: 3-7

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    The loop iterator is 3.
    The loop iterator is 1.
    The loop iterator is 0.
    The loop iterator is 2.
    The loop iterator is 4.
    
Single and master constructs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is sometimes necessary to execute a structured block only once inside a parallel region.
The :code:`single` construct does exactly this:

.. code-block:: c

    #pragma omp single [clause[ [,] clause] ... ] new-line 
        structured-block

The structured block is executed **only once** by **one of the threads** in the team:

.. code-block:: c
    :linenos:
    :emphasize-lines: 4,7
    
    #include <stdio.h>

    int main() {
        #pragma omp parallel
        {
            printf("In parallel.\n");
            #pragma omp single
            printf("Only once.\n");
            printf("More in parallel.\n");
        }
    }

.. code-block:: bash
    :emphasize-lines: 4,9-12

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program                                 
    In parallel.
    Only once.
    In parallel.
    In parallel.
    ...
    In parallel.
    More in parallel.
    More in parallel.
    ...
    More in parallel.

Note that all :code:`In parallel` lines and the :code:`Only once` line are printed before any :code:`More in parallel` lines are printed.
This happens because the :code:`single` construct introduces an **implicit barrier to the exit of the single region**.
That is, all threads in the team must wait until one of the threads has executed the structured block that is associated with the :code:`single` construct:

.. figure:: img/barrier.png
    :align: center
    :scale: 85%

We can disable this behaviour using the :code:`nowait` clause:
    
.. code-block:: c
    :emphasize-lines: 5

    private(list) 
    firstprivate(list) 
    copyprivate(list) 
    allocate([allocator :] list) 
    nowait
    
The :code:`single` construct is closely connected to the :code:`master` construct:

.. code-block:: c

    #pragma omp master new-line 
        structured-block
        
However, there are two primary differences:

 1. Only the **master** thread of the current team can execute the associated structured block.
 2. There is no implied barrier either on entry to, or exit from, the master region.
        
Critical  construct
^^^^^^^^^^^^^^^^^^^

It is sometimes necessary to allow only one thread to execute a structured block concurrently:

.. code-block:: c

    #pragma omp critical [(name) [[,] hint(hint-expression)] ] new-line 
        structured-block

Several :code:`critical` constructs can be joined together by giving them the same name:

.. code-block:: c

    #pragma omp critical (protect_x) 
        x++;
    
    ...
    
    #pragma omp critical (protect_x) 
        x = x - 15;
        
.. challenge::

    Modify the following program such that the :code:`printf` and :code:`number++` statements are protected:

    .. code-block:: c
        :linenos:
        :emphasize-lines: 6

        #include <stdio.h>
        
        int main() {
            int number = 1;
            #pragma omp parallel
            printf("I think the number is %d.\n", number++);
            return 0;
        }

.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 6

        #include <stdio.h>
        
        int main() {
            int number = 1;
            #pragma omp parallel
            #pragma omp critical
            printf("I think the number is %d.\n", number++);
            return 0;
        }
    
    .. code-block:: bash
        :emphasize-lines: 3-6
    
        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program                                 
        I think the number is 1.
        I think the number is 2.
        I think the number is 3.
        I think the number is 4.
        ...
        
Barrier construct
^^^^^^^^^^^^^^^^^

Finally, we can add an **explicit** barrier:

.. code-block:: c

    #pragma omp barrier new-line
    
That is, all threads in the team must wait until all other threads in the team have encountered the :code:`barrier` construct:

.. figure:: img/barrier.png
    :align: center
    :scale: 85%
