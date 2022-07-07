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
