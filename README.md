Task-based parallelism in scientific computing
==============================================

**Abstract:** The purpose of the course is to learn when a code could benefit from task-based parallelism, and how to apply it. A task-based algorithm comprises of a set of self-contained tasks that have well-defined inputs and outputs. This differs from the common practice of organizing an implementation into subroutines in that a task-based implementation does not call the associated computation kernels directly, instead it is the role of a runtime system to schedule the task to various computational resources, such as CPU cores and GPUs. One of the main benefits of this approach is that the underlying parallelism is exposed automatically as the runtime system gradually traverses the resulting task graph.

**Content:** The course mainly focuses on the task-pragmas implemented in the newer incarnations of OpenMP. Other task-based runtime systems, e.g., StarPU, and GPU offloading are briefly discussed.

**Format:** The course will be three half-days and comprises of lectures and hands-on sessions. This is an online-only course (Zoom).

**Audience:** This HPC2N course is part of the PRACE Training courses. It is open for academics and people who work at industry in PRACE member countries.

**Date and Time:** 2021-05-{10,11,12}, 9:00-12:00

**Location:** Online through Zoom

**Instructors:** Mirko Myllykoski (mirkom@cs.umu.se)

**Helpers:** Birgitte Brydsö, Pedro Ojeda-May

**Original author:** Mirko Myllykoski (sprint 2021)

**Prerequisites:**

 - Basic knowledge of C programming language.
 - Basic knowledge of parallel programming.
 - Basic Linux skills.
 - Basic knowledge of OpenMP is beneficial but not required.

**Materials:** https://hpc2n.github.io/Task-based-parallelism/branch/master/

**Recording::: https://www.youtube.com/playlist?list=PL6jMHLEmPVLyVIp67mW1cRj0xbL-6iFMY

**Registration:** https://www.hpc2n.umu.se/events/courses/task-based-parallelism-spring-2021
