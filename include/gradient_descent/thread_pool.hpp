// Inspired by https://github.com/progschj/ThreadPool
#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    ThreadPool(unsigned num_threads = std::max(1u, std::thread::hardware_concurrency() - 1));

    ~ThreadPool();

    template <typename Fn, typename... Args>
    decltype(auto) add_task(Fn&& fn, Args&&... args);

private:
    void run();

    std::queue<std::packaged_task<void()>> tasks;
    std::atomic<bool> is_stopped { false };

    std::vector<std::thread> workers;
    std::condition_variable cv_task;
    std::mutex mutex;
};

template <typename Fn, typename... Args>
decltype(auto) ThreadPool::add_task(Fn&& fn, Args&&... args)
{
    using return_type = typename std::invoke_result<Fn, Args...>::type;

    std::packaged_task<return_type()> task(
        std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));

    std::future<return_type> future = task.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (is_stopped.load()) {
            throw std::runtime_error("thread pool was stopped");
        }
        tasks.emplace(std::move(task));
    }
    cv_task.notify_one();
    return future;
}
