Instructor's guide
------------------

The course is planned to consists of three half days:

:Day 1: Setup, motivation and OpenMP basics

:Day 2: Introduction to OpenMP tasks

:Day 3: StarPU and GPU offloading

sphinx-lesson
^^^^^^^^^^^^^

The course website is built on top of sphinx-lesson: https://coderefinery.github.io/sphinx-lesson/

How to run locally:

 1. Build the materials::

        $ make dirhtml
        ...
        The HTML pages are in _build/dirhtml.
        
 2. Start a local web server::
 
        $ cd _build/dirhtml
        $ python -m SimpleHTTPServer 8080

 3. Open in your browser: http://localhost:8080/

You may have to run `make clean` and restart the server.
This is necessary, for example, if the side menu does not update.

Presentations
"""""""""""""

For presentations, it is beneficial to enter either the **plain mode** or the **presentation mode**:

:Plain mode:            Add :code:`?plain` to the URL.
:Presentation mode:     Add :code:`?minipres` to the URL.

If you are using Google Chrome, you can open the browser in the app mode::

    $ google-chrome --app=https://hpc2n.github.io/Task-based-parallelism/branch/master/
