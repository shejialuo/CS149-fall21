#ifndef _TASKSYS_H
#define _TASKSYS_H

#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <unordered_set>
#include <set>
#include <unordered_map>
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

class Task {
public:
  TaskID id;
  IRunnable* runnable;
  int processing = 0;
  int finished = 0;
  int total_tasks;
  size_t dependencies;
  std::mutex task_mutex;
  Task(TaskID id_, IRunnable* runnable_, int total_tasks_, size_t deps)
    :id(id_), runnable(runnable_), total_tasks(total_tasks_), dependencies(deps) {}
};

class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
private:
  bool terminate = false; // To indicate whether to stop the thread pool
  int _num_threads = 0; // To indicate how many threads
  int sleepThreadNum = 0; // The number of thread which is sleeping
  std::unordered_map<TaskID, Task*> finished {}; // To record the finished task
  std::vector<Task*> ready {}; // The task is ready to be processed
  std::unordered_set<Task*> blocked {}; // The task is blocked
  std::vector<std::thread> threads;
  std::unordered_map<TaskID, std::unordered_set<Task*>> depencency {}; // The depencency information
  TaskID id = 0;
  std::mutex queue_mutex;
  std::condition_variable consumer;
  std::condition_variable producer;
  void start(int num_threads);
  void threadLoop(int index);
  void deleteFinishedTask(Task* task);
  void moveBlockTaskToReady();
  void signalSync();
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
