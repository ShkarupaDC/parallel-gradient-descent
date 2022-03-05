#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "gradient_descent/interface.hpp"

struct SyncState {
    std::condition_variable cv_task;
    std::condition_variable cv_finished;
    std::mutex mutex;
};

class Worker {
public:
    Worker(std::shared_ptr<SyncState> state);

    template <typename Task>
    void set_task(Task&& task);

    void set_finished(bool finished);

    void run();

    void stop();

private:
    mutable std::shared_ptr<SyncState> state;
    std::optional<std::function<void()>> task { std::nullopt };
    std::atomic<bool> finished { false };
    std::thread thread;
};

namespace LinearRegression {

class ParallelSGD : public Interface {
public:
    explicit ParallelSGD(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize, unsigned num_step_epochs = 1, unsigned num_threads = std::thread::hardware_concurrency());

protected:
    std::vector<double> optimize(const Matrix& input, const Matrix& target) override;

private:
    std::vector<std::unique_ptr<Worker>> create_workers() const;

    std::vector<std::pair<Matrix, Matrix>> make_data_chunks(const Matrix& input, const Matrix& target) const;

    std::function<void()> make_task(const Matrix& input, const Matrix& target);

    void distribute_tasks(const std::vector<std::unique_ptr<Worker>>& workers, const std::vector<std::pair<Matrix, Matrix>>& chunks);

    void wait_all_finished() const;

    void update_params();

    double compute_cost(const Matrix& input, const Matrix& target) const;

    void stop_workers(const std::vector<std::unique_ptr<Worker>>& workers) const;

    unsigned num_step_epochs;
    unsigned num_threads;

    mutable std::shared_ptr<SyncState> state;
    std::vector<Params> all_params;
};

}
