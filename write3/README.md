# Assignment 3

## Part 1

The result is illustrated in the following figure.

![saxpy cuda](./assets/saxpy_cuda.png)

### Question 1

As you can see, due to the parallelism of the GPU, the bandwidth is huge.

### Question 2

From the results, we can get that the bottleneck is the memory and CPU of the host itself.
For the execution time of the kernel, the speed is very fast. However, the speed is low
for copying the host memory into the device memory and vice versa.

## Part 2

First, we need to implement the parallelism form of `scan`. The algorithm is super wonderful.
You should first understand the algorithm.

```c
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

It may seem that we need `N` threads for every inner loop, however, this
is a stupid idea, we should calculate how many threads we need for each
inner loop.

The result is below:

![scan cude](./assets/scan_cuda.png)

When dealing with finding repeats. Things would be a little tricky. However,
it is not difficult.

![find repeats cuda](./assets/find_repeats_cuda.png)
