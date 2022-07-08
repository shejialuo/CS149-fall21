# Assignment 1

## Program 1

### Question 1.1

It's easy to split the task.

```c++
int numRows =  args->height / args->numThreads;
int remainRows = args->height % args->numThreads;
int startRow = args->threadId * numRows;
if(args->threadId == args->numThreads - 1 && remainRows != 0) {
  numRows += remainRows;
}
mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                 args->width, args->height, startRow, numRows,
                 args->maxIterations ,args->output);
```

### Question 1.2

The data for view 1 is below:

```txt
# thread_num # speedup
    1            1
    2            1.99
    3            1.65
    4            2.46
    5            2.50
    6            3.28
    7            3.43
    8            4.07
```

And I use gnuplot to plot the figure.

```gp
set title "the speedup of the thread for view 1"
set xlabel "thread number"
set ylabel "speedup"
plot "speedup_view1" w lp
set terminal pngcairo
set output "The speedup for view 1.png"
replot
set terminal qt
set output
```

![The speedup for view 1](./assets/The%20speedup%20for%20view%201.png)

By the way, I also plot the speedup for view 2.

![The speedup for view 2](./assets/The%20speedup%20for%20view%202.png)

As this two picture shows, we can find that the speedup for view 2 is nearly
linear but not for view 1. The reason is the imbalance of the data. For view 1,
some threads handle more data but some handle little, however, we still need to
wait for all threads which wastes lots of time.

### Question 1.3

![The imbalance of the workload for view 1](./assets/The%20imbalance%20of%20the%20workload%20for%20view%201.png)

![The imbalance of the workload for view 2](./assets/The%20imbalance%20of%20the%20workload%20for%20view%202.png)

### Question 1.4

The most important thing to do is to balance the load. But we don't
know the actual load, so we should make slice for each thread job.
Don't let a thread do a consecutive area. But just handle only one raw.

```c++
void workerThreadStartBalance(WorkerArgs * const args) {

    double startTime = CycleTimer::currentSeconds();
    int numRows =  args->height / args->numThreads;
    for(int i = 0;i < numRows; ++i) {
      int startRow = i * args->numThreads + args->threadId;
      mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                       args->width, args->height, startRow, 1,
                       args->maxIterations ,args->output);
    }

    // Here we need to handle some corner case
    int remainRows = args->height % args->numThreads - 1;
    if(remainRows >= 0 && args->threadId <= remainRows) {
      int startRow = numRows * args->numThreads + args->threadId;
      mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                       args->width, args->height, startRow, 1,
                       args->maxIterations ,args->output);
    }

    double endTime = CycleTimer::currentSeconds();
    double interval = endTime - startTime;
    printf("[thread %d]:\t\t[%.3f] ms\n", args->threadId ,interval * 1000);
    printf("Hello world from thread %d\n", args->threadId);
}
```

![The solution to the imbalance](./assets/The%20solution%20to%20the%20imbalance.png)

### Question 1.5

It's not, the reason is because that the *Amdahl's* law. Which
says that if a fraction $r$ of the serial program remains
unparallelizedm we can't get a speedup better than $1/r$. In this example,
we have an important serial function `mandelbrotSerial`, so we can guess the
value of $r$
to be 0.9 (I guess). So no matter how many threads we create,
we could not get a speedup better than 9.

## Program 2

This program is aimed at using SIMD. However, I have learned some basic ideas
in CS61C.

### Question 2.1

```c++
int clampedExpVectorHelper(float* values, int* exponents, float* output, int N) {
  __cs149_vec_float x;
  __cs149_vec_int y;
  __cs149_vec_float result;
  __cs149_vec_int zero = _cs149_vset_int(0);
  __cs149_vec_int one = _cs149_vset_int(1);
  __cs149_vec_float max = _cs149_vset_float(9.999999f);
  __cs149_mask maskAll, maskIsExponentZero;
  __cs149_mask maskIsNotExponentZero, loop, isExceedMax;

  int i = 0 ;
  for(;i + VECTOR_WIDTH <= N; i += VECTOR_WIDTH) {
    maskAll = _cs149_init_ones();
    isExceedMax = _cs149_init_ones(0);

    result = _cs149_vset_float(0.0f);
    _cs149_vload_int(y, exponents + i, maskAll);
    _cs149_veq_int(maskIsExponentZero, y, zero, maskAll);
    _cs149_vset_float(result, 1.f, maskIsExponentZero);

    maskIsNotExponentZero = _cs149_mask_not(maskIsExponentZero);
    _cs149_vload_float(x, values + i, maskIsNotExponentZero);
    _cs149_vadd_float(result, result, x , maskIsNotExponentZero);
    _cs149_vsub_int(y, y, one, maskIsNotExponentZero);
    _cs149_vgt_int(loop, y, zero, maskIsNotExponentZero);
    while(_cs149_cntbits(loop) != 0) {
      _cs149_vmult_float(result, result, x , loop);
      _cs149_vsub_int(y, y, one, loop);
      _cs149_vgt_int(loop, y, zero, loop);
    }
    _cs149_vgt_float(isExceedMax, result, max, maskIsNotExponentZero);
    _cs149_vset_float(result, 9.999999f, isExceedMax);

    _cs149_vstore_float(output + i, result, maskAll);
  }
  return i;
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {

  int index = clampedExpVectorHelper(values, exponents, output, N);

  if(index != N ) {
    float valuesTemp[VECTOR_WIDTH] {};
    int exponentsTemp[VECTOR_WIDTH] {};
    float outputTemp[VECTOR_WIDTH] {};
    for(int i = index; i < N; ++i) {
      valuesTemp[i - index] = values[i];
      exponentsTemp[i - index] = exponents[i];
    }
    clampedExpVectorHelper(valuesTemp, exponentsTemp, outputTemp, VECTOR_WIDTH);
    for(int i = index ; i < N; ++i) {
      output[i] = outputTemp[i - index];
    }
  }
}
```

### Question 2.2

When the vector width is 2:

```txt
Vector Width:              2
Total Vector Instructions: 167727
Vector Utilization:        77.3%
Utilized Vector Lanes:     259325
Total Vector Lanes:        335454
```

When the vector width is 4:

```txt
Vector Width:              4
Total Vector Instructions: 97075
Vector Utilization:        70.2%
Utilized Vector Lanes:     272541
Total Vector Lanes:        388300
```

When the vector width is 8:

```txt
Vector Width:              8
Total Vector Instructions: 52877
Vector Utilization:        66.5%
Utilized Vector Lanes:     281229
Total Vector Lanes:        423016
```

When the vector width is 16:

```txt
Vector Width:              16
Total Vector Instructions: 27592
Vector Utilization:        64.8%
Utilized Vector Lanes:     285861
Total Vector Lanes:        441472
```

As you can see, the vector utilization decreases, the reason I think
is that the branch we make in each vector, when the vector size becomes
bigger, the branch is more likely.

### Question 2.3

It's easy.

```c++
float arraySumVector(float* values, int N) {
  float value[VECTOR_WIDTH] {};
  __cs149_vec_float sum = _cs149_vset_float(0.0f);
  __cs149_vec_float num;
  __cs149_mask maskAll = _cs149_init_ones();
  for (int i=0; i<N; i+=VECTOR_WIDTH) {
    _cs149_vload_float(num, values + i, maskAll);
    _cs149_vadd_float(sum, sum, num, maskAll);
  }
  int number = VECTOR_WIDTH;
  while (number /= 2) {
    _cs149_hadd_float(sum, sum);
    _cs149_interleave_float(sum, sum);
  }
  _cs149_vstore_float(value, sum, maskAll);
  return value[0];
}
```
