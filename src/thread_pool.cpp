#include <stdexcept>

#include <gradient_descent/thread_pool.hpp>

ThreadPool::ThreadPool(unsigned num_threads)
{
    for (unsigned idx = 0; idx < num_threads; ++idx) {
        workers.emplace_back(&ThreadPool::run, this);
    }
}

ThreadPool::~ThreadPool()
{
    is_stopped.store(true);
    cv_task.notify_all();

    for (auto& worker : workers) {
        worker.join();
    }
}

void ThreadPool::run()
{
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->mutex);

            cv_task.wait(lock, [this]() {
                return is_stopped.load() || !tasks.empty();
            });
            if (is_stopped.load() && tasks.empty()) {
                break;
            }
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}
