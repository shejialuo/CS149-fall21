#include "tasksys.h"

#include <iostream>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
  return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads)
  : ITaskSystem(num_threads), _num{num_threads} {
  start(num_threads);
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {

  terminate = true;

  for(auto task : finished) {
    delete task.second;
  }

  producer.notify_all();
  for(int i = 0; i < _num; i++) {
    threads[i].join();
  }
}

void TaskSystemParallelThreadPoolSleeping::start(int num_threads) {
  threads.resize(num_threads);
  for(int i = 0; i < num_threads; ++i) {
    threads[i] = std::move(std::thread(&TaskSystemParallelThreadPoolSleeping::threadLoop, this));
  }
}

void TaskSystemParallelThreadPoolSleeping::threadLoop() {
  while(true) {
    int index = -1;
    {
      std::unique_lock<std::mutex> guard{queue_mutex};
      if(ready.empty()) {
        moveBlockTaskToReady();
        if(ready.empty()) {
          producer.wait(guard);
        }
      }
      /*
        * Here, we must tell whether the ready is empty,
        * when ready.size() == 0, rand() % 0 will cause
        * float point exception. It sucks.
      */
      if(!ready.empty()) index = rand() % ready.size();
    }
    if(terminate) return;
    if(index == -1) continue;
    auto task = ready[index];
    int processing = -1, finished = -1;
    {
      std::unique_lock<std::mutex> guard{task->task_mutex};
      processing = task->processing;
      task->processing++;
    }
    if(processing < task->total_tasks) {
      task->runnable->runTask(processing, task->total_tasks);
      std::unique_lock<std::mutex> guard{task->task_mutex};
      task->finished++;
      finished = task->finished;
    }
    if(finished == task->total_tasks) {
      std::unique_lock<std::mutex> guard{queue_mutex};
      deleteFinishedTask(task, index);
      signalSync();
    }
  }
}

void TaskSystemParallelThreadPoolSleeping::deleteFinishedTask(Task* task, int index) {
  finished.insert({ready[index]->id ,ready[index]});
  ready.erase(ready.begin() + index);
  if(depencency.count(task->id)) {
    for(auto t: depencency[task->id]) {
      t->dependencies--;
    }
  }
}

void TaskSystemParallelThreadPoolSleeping::moveBlockTaskToReady() {
  std::vector<Task*> moved {};
  for(auto task : blocked) {
    if(task->dependencies == 0) {
      ready.push_back(task);
      moved.push_back(task);
    }
  }
  for(auto task: moved) {
    blocked.erase(task);
  }
}

void TaskSystemParallelThreadPoolSleeping::signalSync() {
  if(ready.empty() && blocked.empty()) {
    consumer.notify_one();
  }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
  runAsyncWithDeps(runnable, num_total_tasks, {});
  sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {
  size_t dependencies = 0;
  for(auto dep : deps) {
    if(!finished.count(dep)) {
      dependencies++;
    }
  }
  Task* task = new Task(id, runnable, num_total_tasks, dependencies);
  if(task->dependencies == 0) {
    ready.push_back(task);
  }
  else {
    for(auto dep : deps) {
      if(depencency.count(dep)) {
        depencency[dep].insert(task);
      } else {
        depencency[dep] = std::unordered_set<Task*>{task};
      }
    }
    blocked.insert(task);
  }
  producer.notify_all();
  return id++;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
  std::unique_lock<std::mutex> lock{queue_mutex};
  if(!ready.empty() || !blocked.empty()) {
    consumer.wait(lock);
  }
}
