Introduction to OpenMP
----------------------

.. objectives::

 - Understand the basics of OpenMP

`OpenMP <https://www.openmp.org/>`__ (*Open Multi-Processing*) is a programming API for shared-memory parallel programming in C, C++, and Fortran languages.
It is based on **pragmas** or **directives** which augment the source code and change how a compiler processes the source code.
In case of OpenMP, the pragmas specify how the code is to be parallelized.

.. important::

    The Kebnekaise login nodes have the :code:`OMP_NUM_THREADS` environmental variable set to :code:`1`.
    If you are using the Kebnekaise login nodes to experiment with OpenMP, then it is important to set the :code:`OMP_NUM_THREADS` environmental variable to some reasonable value:
    
    .. code-block:: bash
    
        $ export OMP_NUM_THREADS=8

    Please note that you are not allowed to run long computations on the login nodes!
        
Simple example
^^^^^^^^^^^^^^

Consider the following "Hello world" program:

.. code-block:: c
    :linenos:

    #include <stdio.h>
    
    int main() {
        printf("Hello world!\n");
        return 0;
    }

We can confirm that the code indeed behaves the way we expect:
    
.. code-block:: bash
    :emphasize-lines: 3

    $ gcc -o my_program my_program.c -Wall
    $ ./my_program
    Hello world!

Let us modify the program by adding an OpenMP pragma:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 4

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel
        printf("Hello world!\n");
        return 0;
    }

This time the program behaves very differently (note the extra :code:`-fopenmp` compiler option):
    
.. code-block:: bash
    :emphasize-lines: 3-6

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Hello world!
    Hello world!
    ...
    Hello world!

Clearly, the :code:`omp parallel` pragma caused the program to execute the :code:`printf` line **several times**.
If you go and try to execute the program on a different computer, you will observe that the number of lines printed is the same as the **number of processor cores in the computer**.
The :code:`-fopenmp` compiler option tells the compiler to expect OpenMP pragmas.

.. challenge::

    1. Compile the "Hello world" program yourself and try it out.
    
    2. See what happens if you set the :code:`OMP_NUM_THREADS` environmental variable to different values:
    
    .. code-block:: bash
    
        $ OMP_NUM_THREADS=<value> ./my_program
    
    What happens?
    Can you guess why?
    
.. solution::

    Let us try values 1, 4 and 8:

    .. code-block:: bash
        :emphasize-lines: 2,4-7,9-16

        $ OMP_NUM_THREADS=1 ./my_program
        Hello world!
        $ OMP_NUM_THREADS=4 ./my_program
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        $ OMP_NUM_THREADS=8 ./my_program
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
        Hello world!
    
    The "Hello world!" line is printed 1, 4 and 8 times.
    The :code:`OMP_NUM_THREADS` environmental variable sets the default team size (see below).

OpenMP pragmas and constructs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In C and C++, an OpenMP pragma has the following form:

.. code-block:: c

    #pragma omp directive-name [clause[ [,] clause] ... ] new-line

A compiler typically supports several types of pragmas, not just OpenMP pragmas.
Therefore, all OpenMP pragmas begin with the keywords :code:`#pragma omp`.
The :code:`directive-name` placeholder specifies the used OpenMP construct (e.g. :code:`parallel`) and a pragma is always followed by a new line.
Typically, a pragma affects the user code that follows it but some OpenMP pragmas are *stand-alone*.
You can span a pragma across multiple lines by using a backslash (:code:`\ `) immediately followed by a new line:

.. code-block:: c

    #pragma omp directive-name \
        [clause[ [,] \
        clause] ... ] new-line
        
Parallel construct
^^^^^^^^^^^^^^^^^^

In the earlier example, we used the :code:`parallel` pragma:

.. code-block:: c

    #pragma omp parallel [clause[ [,] clause] ... ] new-line 
        structured-block

The pragma creates a **team** of **OpenMP threads** that executes the :code:`structured-block` region:

.. figure:: img/parallel_construct.png
    :align: center
    :scale: 75%

The :code:`structured-block` region can be a single statement, like in the earlier example, or a structured block consisting of several statements: 

.. code-block:: c

    #pragma omp parallel ...
    {
        statement1;
        statement2;
        ...
    }

OpenMP guarantees that all threads in the team have executed the structured block before the execution continues outside the parallel construct. 
    
The behaviour of a parallel construct can be modified with several **clauses**:

.. code-block:: bash
    :emphasize-lines: 1-2

    if([parallel :] scalar-expression) 
    num_threads(integer-expression) 
    default(shared | none) 
    private(list) 
    firstprivate(list) 
    shared(list) 
    copyin(list) 
    reduction([reduction-modifier ,] reduction-identifier : list) 
    proc_bind(master | close | spread) 
    allocate([allocator :] list)

We will return to some of these clauses later but for now it is sufficient to know that a parallel construct can be selectively enabled/disabled with the :code:`if` clause and the size of the team can be explicitly set with the :code:`num_threads` clause.

.. challenge::

    Modify the following program such that the :code:`printf` line is executed only twice:
    
    .. code-block:: c
        :linenos:

        #include <stdio.h>
    
        int main() {
            #pragma omp parallel
            printf("Hello world!\n");
            return 0;
        }
    
    **Hint:** Each thread in the team executes the structured block once.

.. solution::

    Use the :code:`num_threads` clause to set the team size to two:

    .. code-block:: c
        :linenos:
        :emphasize-lines: 4

        #include <stdio.h>
    
        int main() {
            #pragma omp parallel num_threads(2)
            printf("Hello world!\n");
            return 0;
        }
    
    .. code-block:: bash
        :emphasize-lines: 3-4

        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        Hello world!
        Hello world!

Data sharing rules
^^^^^^^^^^^^^^^^^^

Since the structured block that follows a parallel construct is executed in parallel by a team of threads, we must make sure that the related data accesses do not cause any **conflicts**.
For example, the behaviour of the following program is not well defined:

.. code-block:: c
    :linenos:

    #include <stdio.h>
    
    int main() {
        int number = 1;
        #pragma omp parallel
        printf("I think the number is %d.\n", number++);
        return 0;
    }

.. code-block:: bash
    :emphasize-lines: 3,6,7,9,12,13,14,16,17

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    I think the number is 2.
    I think the number is 8.
    ....
    I think the number is 1.
    I think the number is 1.
    ....
    I think the number is 2.
    ....
    $ ./my_program 
    I think the number is 1.
    I think the number is 1.
    I think the number is 2.
    ...
    I think the number is 1.
    I think the number is 2.
    ...

We can make two observations:

 1. The order in which the :code:`printf` statements are executed is arbitrary. This can be a desired behaviour.
 2. Some numbers are printed **multiple times**. This is usually an undesired behaviour.

The explanation is that once the team is created, the threads execute the structured block **independently** of each other.
This explain why the numbers are printed in an arbitrary order.
The threads also read and write the variable :code:`number` independently of each other which explain why some threads do not see the changes the other threads have made:

.. figure:: img/conflict.png
    :align: center
    :scale: 75%

OpenMP implements a set of rules that define how variables behave inside OpenMP constructs.
All variables are either :code:`private` or :code:`shared`:

:Private:   Each thread has its own copy of the variable.
:Shared:    All threads share the same variable.

These basic **rules** apply:

 1. All variables declared outside parallel constructs are shared.
 2. All variables declared inside a parallel construct are private.
 3. Loop counters are private (in parallel loops).

.. code-block:: c
    :linenos:
    :emphasize-lines: 1,4,8
    
    int a = 5;                  // shared
    
    int main() {
        int b = 44;             // shared
        
        #pragma omp parallel
        {
            int c = 3;          // private
        }
    }

In the above example, the variable :code:`number` is declared outside the parallel construct and all threads therefore share the same variable.

.. challenge::

    Modify the following program such that the variable :code:`number` is declared inside the structured block and is therefore private:
    
    .. code-block:: c
        

        #include <stdio.h>
    
        int main() {
            int number = 1;
            #pragma omp parallel
            printf("I think the number is %d.\n", number++);
            return 0;
        }
    
    Run the program.
    Can you explain the behaviour?
    
    **Hint:** Remember that structured block that consists of several statements must be enclosed inside :code:`{ }` brackets. 

.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 5-8

        #include <stdio.h>

        int main() {
            #pragma omp parallel
            {
                int number = 1;
                printf("I think the number is %d.\n", number++);
            }
            return 0;
        }
        
    .. code-block:: bash
        :emphasize-lines: 3-6

        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        I think the number is 1.
        I think the number is 1.
        ...
        I think the number is 1.
    
    Note that all treads print 1.
    This happens because each thread has its own :code:`number` variable that is initialized to 1.
    The incrementation affects only the thread's own copy of the variable.

We can use the **private** clause to turn a variable that has been declared outside a parallel construct into a private variable:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5

    #include <stdio.h>
    
    int main() {
        int number = 1;
        #pragma omp parallel private(number)
        printf("I think the number is %d.\n", number++);
        return 0;
    }

However, the end result is, once again, unexpected:

.. code-block:: bash
    :emphasize-lines: 3-6

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    I think the number is 0.
    I think the number is 0.
    I think the number is 0.
    ...

This happens because each thread has its own :code:`number` variable that is separate from the :code:`number` variable declared outside the parallel construct:

.. figure:: img/private.png
    :align: center
    :scale: 75%

The private variables do **not inherit the value of the original variable**.
If we want this to happen, then we must use the **firstprivate** clause:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5

    #include <stdio.h>
    
    int main() {
        int number = 1;
        #pragma omp parallel firstprivate(number)
        printf("I think the number is %d.\n", number++);
        return 0;
    }

This time, the end result is as expected:

.. code-block:: bash
    :emphasize-lines: 3-6

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    I think the number is 1.
    I think the number is 1.
    I think the number is 1.
    ...

That is, the private variables inherits the value of the original variable:

.. figure:: img/firstprivate.png
    :align: center
    :scale: 75%

Explicit data sharing rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default behaviour can be changed with the **default** clause:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5

    #include <stdio.h>
    
    int main() {
        int number = 1;
        #pragma omp parallel default(none)
        printf("I think the number is %d.\n", number++);
        return 0;
    }

This tells the compiler that a programmer must explicitly set the data sharing rule for each variable.
It is therefore not surprising that the compiler produces an error indicating that the :code:`number` variable is not specified in the enclosing parallel construct:

.. code-block:: bash
    :emphasize-lines: 2-8

    $ gcc -o my_program my_program.c -Wall -fopenmp 
    my_program.c: In function ‘main’:
    my_program.c:6:5: error: ‘number’ not specified in enclosing ‘parallel’
        6 |     printf("I think the number is %d.\n", number++);
          |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    my_program.c:5:13: error: enclosing ‘parallel’
        5 |     #pragma omp parallel default(none)
          |

We can now set the :code:`number` variable to first private:

.. code-block:: c
    :linenos:
    :emphasize-lines: 5

    #include <stdio.h>
    
    int main() {
        int number = 1;
        #pragma omp parallel default(none) firstprivate(number)
        printf("I think the number is %d.\n", number++);
        return 0;
    }

It is **generally recommended** that a programmer sets the data sharing rules explicitly as this forces them to think about the data sharing rules.
It is also advisable to declare all private variables inside the structured block.

.. challenge::

    Fix the following program:
    
    .. code-block:: c
        :linenos:

        #include <stdio.h>

        char *str = "I think the number is %d.\n";

        int main() {
            int initial_number = 1;

            #pragma omp parallel
            int number = initial_number; 
            printf(str, number++);
    
            return 0;
        }

    Use explicit data sharing rules.

.. solution::

    .. code-block:: c
        :linenos:
        :emphasize-lines: 8,9,12
        
        #include <stdio.h>

        char *str = "I think the number is %d.\n";

        int main() {
            int initial_number = 1;

            #pragma omp parallel default(none) shared(str, initial_number)
            {
                int number = initial_number; 
                printf(str, number++);
            }
    
            return 0;
        }
         
    First, we add the enclosed :code:`{ }` brackets thus making the :code:`number` variable private.
    Next, we use :code:`default(none)` to force explicit data sharing rules.
    Finally, we declare the :code:`str` and :code:`initial_number` variables shared as none of the threads modify these variables.
    
    .. code-block:: bash

        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        I think the number is 1.
        I think the number is 1.
        I think the number is 1.
        ...
    
    It is also possible to declare the variables :code:`str` and :code:`initial_number` as :code:`firstprivate`.
    However, the creation of private variables causes some overhead and it is therefore generally recommended that variables that can be declared shared are declared as shared.

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
        
The construct tells OpenMP that the loop iterations are free of data dependencies and can therefore executed in parallel.
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
        
    Note that the iterations from the both loops are now executed in an arbitrary order.

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

It is sometimes necessary to execute a structured block only once inside a parallel construct.
The :code:`single` construct does exactly this:

.. code-block:: c

    #pragma omp single [clause[ [,] clause] ... ] new-line 
        structured-block

The structured block is executed **only once** by **one of the treads** in the team:

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
This happens because the :code:`single` construct introduces an **implicit barrier to the exit of the construct**.
That is, all threads in the team must wait until one of the treads has executed the structured block that is associated with the :code:`single` construct:

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
 2. There is no implied barrier either on entry to, or exit from, the master construct.
        
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
    
That is, all threads in the team must wait until all other threads in the team have encountered the :code:`barrier` construct.
