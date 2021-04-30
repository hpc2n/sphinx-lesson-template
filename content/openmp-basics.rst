Introduction to OpenMP
----------------------

.. objectives::

 - Understand the basics of OpenMP

`OpenMP <https://www.openmp.org/>`__ (*Open Multi-Processing*) is a programming API for shared-memory parallel programming in C, C++, and Fortran languages.
It is based on **pragmas** or **directives** which augment the source code and change how a compiler processes the source code.
In case of OpenMP, the pragmas specify how the code is to be parallelized.

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

Lets modify the program by adding an OpenMP pragma:
    
.. code-block:: c
    :linenos:
    :emphasize-lines: 4

    #include <stdio.h>
    
    int main() {
        #pragma omp parallel
        printf("Hello world!\n");
        return 0;
    }

This time the program behaves very differently (note the extra `-fopenmp` argument):
    
.. code-block:: bash
    :emphasize-lines: 3-6

    $ gcc -o my_program my_program.c -Wall -fopenmp
    $ ./my_program 
    Hello world!
    Hello world!
    ...
    Hello world!

Clearly, the `omp parallel` pragma caused the program to execute the `printf` line **several times**.
If you go and try to execute the program on a different computer, you will observe that the number of lines printed is the same as the number of processor cores in the system.

.. challenge::

    1. Compile the "Hello world" program yourself and try it out.
    
    2. See what happens if you set the `OMP_NUM_THREADS` environmental variable to different values:
    
    .. code-block:: bash
    
        $ OMP_NUM_THREADS=value ./my_program
    
    What happens? Why?
    
.. solution::

    Lets try values 1, 4 and 8:

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
    The `OMP_NUM_THREADS` environmental variable sets the default team size (see below).

OpenMP pragmas and constructs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In C and C++, an OpenMP pragma has the following form:

.. code-block:: c

    #pragma omp directive-name [clause[ [,] clause] ... ] new-line

A compiler typically support several types of pragmas, not just OpenMP pragmas.
Therefore, all OpenMP pragmas begin with `#pragma omp ...`.
The `directive-name` specifies the used OpenMP construct (e.g. `parallel`) and a pragma is always followed by a new line.
Typically, a pragma effects the user code that follows it but some OpenMP pragmas are *stand-alone*.
You can span a pragma across multiple lines by using a backslash (`\\`) immediately followed by a new line:

.. code-block:: c

    #pragma omp directive-name \
        [clause[ [,] \
        clause] ... ] new-line
        
Parallel construct
""""""""""""""""""

In the earlier example, we used the `parallel` pragma:

.. code-block:: c

    #pragma omp parallel [clause[ [,] clause] ... ] new-line 
        structured-block

The pragma creates a **team** of **OpenMP threads** that execute the `structured-block` region.
The `structured-block` region can be a single statement, like in the earlier example, or a structured block consisting from several statements:

.. code-block:: c

    #pragma omp parallel ...
    {
        stetement1;
        statement2;
        ...
    }

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

We will return to some of these clauses later but for now it is sufficient to know that a parallel construct can selectively enabled/disabled with the **if** clause and the size of the team can be explicitly set with the **num_threads** clause.

.. challenge::

    Modify the following program such that the `printf` line is executed only twice:
    
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

    Use the `num_threads` clause to set the team size to two:

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
""""""""""""""""""

Since the structured block that follows a parallel construct is executed in parallel by a team of threads, we must make sure that the related data accesses do cause any conflicts.
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

 1. The order in which the `printf` statements are executed is arbitrary. This can be a desired behaviour.
 2. Some number are printed **multiple times**. This is usually an undesired behaviour.

The explanation is that once the team is created, the threads execute the structured block **independently** of each other.
This explain why the numbers are printed in an arbitrary order.
The threads also read and write the variable `number` independently of each other which explain why some threads do not see the changes the other threads have made.

OpenMP implements a set of rules that define how variables behave inside OpenMP constructs.
All variables are ether

 - **private**, i.e., each thread has its own copy of the variable, or
 - **shared**, i.e., all threads share the same variable.

These basic **rules** apply:

 1. All variables declared outside parallel constructs are shared.
 2. All variables declared inside a parallel construct are private.
 3. Loop counters are private (in parallel loops).

In the above example, the variable `number` is declared outside the parallel construct and all threads therefore share the same variable.

.. challenge::

    Modify the following program such that the variable `number` is declared inside the structured block and is therefore private:
    
    .. code-block:: c
        :linenos:

        #include <stdio.h>
    
        int main() {
            int number = 1;
            #pragma omp parallel
            printf("I think the number is %d.\n", number++);
            return 0;
        }
    
    Run the program.
    Can you explain the behaviour?
    
    **Hint:** Remember that structured block that consists of several statements must be enclosed inside `{ }` brackets. 

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
    This happens because each thread has its own `number` variable that is initialized to 1.
    The incrementation effects only thread's own copy of the variable.

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
    I think the number of 0.
    I think the number of 0.
    I think the number of 0.
    ...

This happens because each thread has its own `number` variable that is separate from the `number` variable declared outside the parallel construct.
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
    I think the number of 1.
    I think the number of 1.
    I think the number of 1.
    ...

That is, the private variables inherit the value of the original variable.

Explicit data sharing rules
"""""""""""""""""""""""""""

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
It is not therefore surprising that the compiler produces an error indicating the the `number` variable is not specified in enclosing parallel construct:

.. code-block:: bash
    :emphasize-lines: 2-8

    $ gcc -o my_program my_program.c -Wall -fopenmp 
    my_program.c: In function ‘main’:
    my_program.c:6:5: error: ‘number’ not specified in enclosing ‘parallel’
        6 |     printf("I think the number of %d.\n", number++);
          |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    my_program.c:5:13: error: enclosing ‘parallel’
        5 |     #pragma omp parallel default(none)
          |

We can not set the `number` variable to first private:

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

        char *str = "I think the number of %d.\n";

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

        char *str = "I think the number of %d.\n";

        int main() {
            int initial_number = 1;

            #pragma omp parallel default(none) shared(str, initial_number)
            {
                int number = initial_number; 
                printf(str, number++);
            }
    
            return 0;
        }
         
    First, we add the enclosed `{ }` brackets thus making the `number` variable private.
    Next, we use `default(none)` to force explicit data sharing rules.
    Finally, we declare the `str` and `initial_number` variables shared as none of the threads modify these variables.
    
    .. code-block:: bash

        $ gcc -o my_program my_program.c -Wall -fopenmp
        $ ./my_program 
        I think the number of 1.
        I think the number of 1.
        I think the number of 1.
        ...
    
    It is also possible to declare the variables `str` and `initial_number` as `firstprivate`.
    However, the creation of private variables causes some overhead and it therefore generally recommended that variables that can be declared shared are declared as shared.

Parallel loop construct
"""""""""""""""""""""""

Section construct
"""""""""""""""""

Single and master constructs
""""""""""""""""""""""""""""

Critical and atomic constructs
""""""""""""""""""""""""""""""

Barrier construct
"""""""""""""""""
