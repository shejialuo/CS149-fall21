# Extra Credit: Implement Dense Matrix-Matrix Multiplication as Fast as You Can #

**Due Fri Dec 3rd, 5:00pm PT (no late submission allowed)**

If you complete this assignment, you will __receive up__ to 10 bonus points on one of the regular programming assignments (PA1-PA4). Note that programming assignment averages are not capped, so this is essentially "extra credit" on the course.  

## Overview ##

In this assignment you will implement general dense matrix-matrix multiplication (GEMM).  GEMM is a critical computational primitive in many applications ranging from linear algebra, solving systems of equations, scientific computing, and modern [deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/).  

## Environment Setup ##

__You will need to run code on the myth machines for this assignment.__
(Hostnames for these machines are `myth[51-66].stanford.edu`)
These machines contain four-core 4.2 GHz Intel Core i7 processors (although dynamic frequency scaling can take them to 4.5 GHz when the chip decides it is useful and possible to do so). Each core in the processor can execute AVX2 vector instructions which describe
simultaneous execution of the same operation on multiple single-precision data
values. For the curious, a complete specification for this CPU can be found at 
<https://ark.intel.com/products/97129/Intel-Core-i7-7700K-Processor-8M-Cache-up-to-4-50-GHz->.

Note: For grading purposes, we expect you to report on the performance of code run on the Stanford myth machines, however
for kicks, you may also want to run the programs in this assignment on your own machine or on AWS machines if you have credits left over.
Feel free to include your findings from running code on other machines in your report as well, just be very clear what machine you were running on. 

To get started:

1. The assignment starter code is available on [github]. Please clone the extra credit starter code using:

    `git clone git@github.com:stanford-cs149/extracredit_gemm`

2. __Note: You can skip this step and still do the assignment, you just won't be able to compare your performance to a professional implementation.__ The Intel Math Kernel Library (MKL) is a highly optimized linear algebra library which you can test your matrix-multiply routine against.  Approaching the performance of this library with a solutions would be an extremely strong effort. You can install MKL on the myth machines using the following steps.  

```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18236/l_BaseKit_p_2021.4.0.3422.sh

bash l_BaseKit_p_2021.4.0.3422.sh
```
Running this script is going to bring up a text based installer. You'll want to deselect everything except for "the Intel Math Kernel Library", and your final installation should take up 7.7GB of space on Myth.

Or, for a one-line installation once you've downloaded you can use this command. Note that you will accept the EULA by running this command.
```
bash l_BaseKit_p_2021.4.0.3422.sh -a --action install --components intel.oneapi.lin.mkl.devel -s --eula accept
```

If that fails, you can manually download MKL as part of the Intel oneAPI Base Toolkit here: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Once you've installed MKL successfully, you'll want to define the `MKLROOT` environment variable to point it at your MKL installation. Once you've done so, the Makefile should detect it and automatically include MKL in any further builds. Note that you will need to `make clean` first. If you've installed MKL to the default directory, you can use the following command:

```bash
export MKLROOT=${HOME}/intel/oneapi/mkl/latest/
```

3. Build the starter code by typing `make` in the `/gemm` directory.  Note that you will need the ISPC compiler in your path. (You did this for programming assignment 1... please see those instructions.)

4. Run the starter code in the `/gemm` directory using the command `./gemm N`.  The parameter `N` specifies the sizes of all dimensions of the matrices.  In other words, if N=1024, then the code will create 1024 x 1024 matrices A, B, and C, and compute `C = alpha * A * B + beta C` (where alpha and beta are scalars). If you have MKL installed, you should see the following output, which documents the performance of three different implementations: Intel MKL library's performance, a staff reference implementation written in ISPC (this is reasonably optimized, (but not close to MKL), and your solution.  The program runs each configuration three times and, for each configuration, reports the fastest of the three runs.  Note how much faster MKL is than the starter code we give you, which is just a simple triple for loop (the best MKL run is ~180 times faster than the best starter code run!!!). 

```
(base) foo@myth55:~/starter/gemm$ ./gemm 1024
Running Intel MKL... 27.94ms
Running student GEMM... 1708.95ms
Running ref ispc GEMM... 113.31ms
Running Intel MKL... 9.62ms
Running student GEMM... 1709.59ms
Running ref ispc GEMM... 112.75ms
Running Intel MKL... 9.58ms
Running student GEMM... 1711.41ms
Running ref ispc GEMM... 112.40ms
[Intel MKL]:        [9.585] ms      [3.260] GB/s    [224.05] GFLOPS
[Student GEMM]:     [1708.952] ms   [0.018] GB/s    [1.26] GFLOPS
[Ref ISPC GEMM]:    [112.398] ms    [0.278] GB/s    [19.11] GFLOPS
Total squared error student sol: 0.000000
Total squared error ref ispc: 0.000000
```
### What you need to do:

Implement GEMM on square (NxN) matrices. __(Your solution can assume all matrices are square with power-of-two dimensions.)__. You can implement your solution in plain C (perhaps with vector instrinsics), or use ISPC.  

* If you want to implement the solution completing from scratch in C + AVX instrinsics, place your solution in `gemm/gemm.cpp`.  You'll find a naive solution already given to you in that file.
* If you want to implement your solution in ISPC, please place it in `gemm/gemm.ispc`.  __You'll need to modify `main.cpp` to call `ispc::gemm_ispc(m, n, k, A2, B2, C2, alpha, beta);`__ instead of `gemm(m, n, k, A2, B2, C2, alpha, beta);` so that the harness runs your ISPC code instead of the C code solution.

That's it!  This is "going further" extra credit, so you are on your own!  We do first recommend ["blocking" (aka tiling)](https://cs149.stanford.edu/fall21/lecture/perfopt2/slide_54) the loops so that you dramatically improve the cache behavior of the algorithm for larger N.  Specifically, how can you decompose a large matrix multiplication of NxN matrices, written using three for loops, into a sequence of smaller KxK matrix multiplications where KxK submatrices of A, B, and C are sized to fit in cache? (this will be a six for-loop implementation).  Once you get your tiling scheme working, then I'd recommend considering multi-core and SIMD parallelization. You could even consider multi-level blocking, e.g., for registers, and different levels of the cache.

For a better understanding of blocking, I recommend you [take a look at this article](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf).  Just doing blocking of the assignment will not get you extra credit, you need to go further than that.  However, you will need to block an implementation as the first major performance improvement.

### Grading

__This extra credit will be graded on a case-by-case basis.__  There is no grading harness, however we are looking for __a writeup that documents a serious effort to obtain improved performance using ideas like creating parallel work for all cores, blocking to fit in cache (and perhaps even registers), SIMD vector processing (e.g. using ISPC or vector instrinsics).__ To give you a ball park estimate of expectations, we are looking for student work to be approaching that of the reference ISPC implementation to gain some extra credit, but don't necessarily expect to beat it to get points -- but you certainly can!).  

Although it's on a completely different topic (optimizing memory usage in Go), recently your instructor was impressed by [this writeup](https://www.akitasoftware.com/blog-posts/taming-gos-memory-usage-or-how-we-avoided-rewriting-our-client-in-rust) that documents a sequence of performance debugging steps.  This is the type of run-observe-improve loop we're looking for you to carry out.

## Hand-in Instructions ##

Handin will be performed via [Gradescope](https://www.gradescope.com/). Please place the following files in your handin:

1. Your implementation of `gemm.cpp` and/or `gemm.ispc`
2. A writeup with:
  1. A graph of GFlops for MKL, your implementation, and the reference ISPC implementation with input argument N of 256, 512, 1024, 2048, and 4096 
  2. A detailed description of the steps you performed in the optimzation process. In particular, please let us know which steps had the biggest benefit.
