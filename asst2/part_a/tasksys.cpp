#include "tasksys.h"


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

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
  : ITaskSystem(num_threads), _num_threads{num_threads} {}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
  auto thread_func = [runnable_ = runnable, num = _num_threads, total = num_total_tasks](int i) {
    while(i < total) {
      runnable_->runTask(i, total);
      i += num;
    }
  };
  std::thread threads[_num_threads];
  for (int i = 0; i < _num_threads; ++i) {
    threads[i] = std::move(std::thread(thread_func, i));
  }
  for (int i = 0; i < _num_threads; ++i) {
    threads[i].join();
  }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
  return 0;
}

void TaskSystemParallelSpawn::sync() {
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads)
  : ITaskSystem(num_threads), _num_threads(num_threads) {
  unsigned int init = 0x01;
  for(int i = 0; i < _num_threads; ++i) {
    bitmap_init_value |= init;
    init <<= 1;
  }
  start(_num_threads);
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
  /*
    * Here, we don't need to synchronize the code, because
    * the thread will never write `terminate`. No matter
    * the thread may read some corrupted value, this doesn't matter.
  */
  terminate = true;
  for(int i = 0; i < _num_threads; ++i) {
    threads[i].join();
  }
}

void TaskSystemParallelThreadPoolSpinning::start(int num_threads) {
  threads.resize(num_threads);
  for(int i = 0; i < num_threads; ++i) {
    threads[i] = std::move(std::thread(&TaskSystemParallelThreadPoolSpinning::threadLoop, this, i));
  }
}

void TaskSystemParallelThreadPoolSpinning::threadLoop(int i) {
  while(true && !terminate) {
    bool flag = false;
    {
      std::lock_guard<std::mutex> guard{queue_mutex};
      flag = (jobs >> i) & 0x01;
    }
    if(flag) {
      int taskId = i;
      while(taskId < total_tasks) {
        runnable_->runTask(taskId, total_tasks);
        taskId += _num_threads;
      }
      {
        std::lock_guard<std::mutex> guard{queue_mutex};
        jobs &= ~(0x01 << i);
      }
    }
  }
}

bool TaskSystemParallelThreadPoolSpinning::busy() {
  bool poolbusy;
  {
    std::lock_guard<std::mutex> guard{queue_mutex};
    poolbusy = jobs;
  }
  return poolbusy;
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
  total_tasks = num_total_tasks;
  runnable_ = runnable;
  {
    std::lock_guard<std::mutex> guard{queue_mutex};
    jobs = bitmap_init_value;
  }
  while(busy());
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
  return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
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
  : ITaskSystem(num_threads), _num_threads(num_threads) {
  unsigned int init = 0x01;
  for(int i = 0; i < _num_threads; ++i) {
    bitmap_init_value |= init;
    init <<= 1;
  }
  start(_num_threads);
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
  /*
    * Here, we don't need to synchronize the code, because
    * the thread will never write `terminate`. No matter
    * the thread may read some corrupted value, this doesn't matter.
  */
  terminate = true;
  producer.notify_all();
  for(int i = 0; i < _num_threads; ++i) {
    threads[i].join();
  }
}

void TaskSystemParallelThreadPoolSleeping::start(int num_threads) {
  threads.resize(num_threads);
  for(int i = 0; i < num_threads; ++i) {
    threads[i] = std::move(std::thread(&TaskSystemParallelThreadPoolSleeping::threadLoop, this, i));
  }
}

void TaskSystemParallelThreadPoolSleeping::threadLoop(int i) {
  while(true) {
    {
      std::unique_lock<std::mutex> guard{queue_mutex};
      if(((jobs >> i) & 0x01)== 0) {
        producer.wait(guard);
      }
    }
    if(terminate) return;
    int taskId = i;
    while(taskId < total_tasks) {
      runnable_->runTask(taskId, total_tasks);
      taskId += _num_threads;
    }
    {
      std::lock_guard<std::mutex> guard{queue_mutex};
      jobs &= ~(0x01 << i);
      if(jobs == 0) {
        consumer.notify_one();
      }
    }
  }
}

bool TaskSystemParallelThreadPoolSleeping::busy() {
  bool poolbusy;
  {
    std::lock_guard<std::mutex> guard{queue_mutex};
    poolbusy = jobs;
  }
  return poolbusy;
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
  total_tasks = num_total_tasks;
  runnable_ = runnable;
  {
    std::lock_guard<std::mutex> guard{queue_mutex};
    jobs = bitmap_init_value;
  }
  producer.notify_all();
  {
    std::unique_lock<std::mutex> guard{queue_mutex};
    if(jobs != 0) {
      consumer.wait(guard);
    }
  }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {
  return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {
  return;
}
