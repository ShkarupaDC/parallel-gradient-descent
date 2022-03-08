#include <stdexcept>

#include <gradient_descent/thread_pool.hpp>

ThreadPool::ThreadPool(unsigned num_threads)
{
    for (unsigned idx = 0; idx < num_threads; ++idx) {
        this->workers.emplace_back(&ThreadPool::run, this);
    }
}

ThreadPool::~ThreadPool()
{
    this->is_stopped.store(true);
    this->cv_task.notify_all();

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

            this->cv_task.wait(lock, [this]() {
                return this->is_stopped.load() || !this->tasks.empty();
            });
            if (this->is_stopped.load() && this->tasks.empty()) {
                break;
            }
            task = std::move(this->tasks.front());
            this->tasks.pop();
        }
        task();
    }
}
