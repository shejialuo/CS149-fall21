#ifndef _TASKSYS_H
#define _TASKSYS_H

#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "itasksys.h"

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
public:
  TaskSystemSerial(int num_threads);
  ~TaskSystemSerial();
  const char* name();
  void run(IRunnable* runnable, int num_total_tasks);
  TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                          const std::vector<TaskID>& deps);
  void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
  private:
    int _num_threads;
  public:
    TaskSystemParallelSpawn(int num_threads);
    ~TaskSystemParallelSpawn();
    const char* name();
    void run(IRunnable* runnable, int num_total_tasks);
    TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps);
    void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
private:
  int _num_threads; // to store the threads
  std::vector<std::thread> threads; // thread poll
  unsigned int jobs = 0x00; // bitmap value for indicating whether there is a job
  unsigned int bitmap_init_value = 0x00; // initialized bitmap value with 0x1111
  IRunnable* runnable_; // we need to record the runnable
  std::mutex queue_mutex; // the big lock
  bool terminate = false; // Whether we should terminate the thread
  int total_tasks = 0;    // we should record the total task
  void start(int num_threads); // start the thread pool
  void threadLoop(int i); // thread functionaility
  bool busy(); // whether the threads are busy doing their jobs
public:
  TaskSystemParallelThreadPoolSpinning(int num_threads);
  ~TaskSystemParallelThreadPoolSpinning();
  const char* name();
  void run(IRunnable* runnable, int num_total_tasks);
  TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                          const std::vector<TaskID>& deps);
  void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
private:
  int _num_threads; // to store the threads
  std::vector<std::thread> threads; // thread poll
  unsigned int jobs = 0x00; // bitmap value for indicating whether there is a job
  unsigned int bitmap_init_value = 0x00; // initialized bitmap value with 0x1111
  IRunnable* runnable_; // we need to record the runnable
  std::mutex queue_mutex; // the big lock
  std::condition_variable consumer; // the condition variable
  std::condition_variable producer; // the condition variable
  bool terminate = false; // Whether we should terminate the thread
  int total_tasks = 0;    // we should record the total task
  void start(int num_threads); // start the thread pool
  void threadLoop(int i); // thread functionaility
  bool busy(); // whether the threads are busy doing their jobs
public:
  TaskSystemParallelThreadPoolSleeping(int num_threads);
  ~TaskSystemParallelThreadPoolSleeping();
  const char* name();
  void run(IRunnable* runnable, int num_total_tasks);
  TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
  void sync();
};

#endif
