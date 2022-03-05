#include <algorithm>
#include <cmath>
#include <random>

#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "gradient_descent/parallel_sgd.hpp"

Worker::Worker(std::shared_ptr<SyncState> state)
    : state(state)
    , thread(std::thread(&Worker::run, this))
{
}

template <typename Task>
void Worker::set_task(Task&& task)
{
    std::lock_guard<std::mutex> lock(this->state->mutex);
    this->task.emplace(std::forward<Task>(task));
};

void Worker::set_finished(bool finished)
{
    this->finished.store(finished);
}

void Worker::run()
{
    while (true) {
        std::unique_lock<std::mutex> lock(this->state->mutex);

        this->state->cv_task.wait(lock, [this]() {
            return this->finished.load() || this->task.has_value();
        });
        if (this->finished.load()) {
            break;
        } else {
            lock.unlock();
            this->task.value()();

            lock.lock();
            this->task.reset();
            this->state->cv_finished.notify_one();
        }
    }
}

void Worker::stop()
{
    this->thread.join();
}

namespace LinearRegression {

ParallelSGD::ParallelSGD(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize, unsigned num_step_epochs, unsigned num_threads)
    : Interface(num_epochs, learning_rate, weight_decay, normalize)
    , num_step_epochs(num_step_epochs)
    , num_threads(num_threads)
    , state(std::make_shared<SyncState>())
{
}

std::vector<std::unique_ptr<Worker>> ParallelSGD::create_workers() const
{
    std::vector<std::unique_ptr<Worker>> workers;
    workers.reserve(this->num_threads);

    for (unsigned idx = 0; idx < this->num_threads; ++idx) {
        workers.push_back(std::make_unique<Worker>(this->state));
    }
    return workers;
}

std::vector<std::pair<Matrix, Matrix>> ParallelSGD::make_data_chunks(const Matrix& input, const Matrix& target) const
{
    int num_samples = input.shape(0);
    int split_size = ceil(num_samples / (float)this->num_threads);

    std::vector<std::pair<Matrix, Matrix>> data;
    data.reserve(this->num_threads);

    for (unsigned idx = 0; idx < this->num_threads; ++idx) {
        int start = idx * split_size;
        int end = std::min(start + split_size, num_samples);

        auto data_split = std::make_pair(
            xt::view(input, xt::range(start, end), xt::all()), xt::view(target, xt::range(start, end), xt::all()));
        data.push_back(std::move(data_split));
    }
    return data;
}

std::function<void()> ParallelSGD::make_task(const Matrix& input, const Matrix& target)
{
    return [&]() {
        auto new_core = this->core;

        for (unsigned epoch = 0; epoch < this->num_step_epochs; ++epoch) {
            new_core.optimize_step(
                input, new_core.compute_prediction(input), target);
        }
        std::lock_guard<std::mutex> lock(this->state->mutex);
        this->all_params.push_back(new_core.get_params());
    };
};

void ParallelSGD::distribute_tasks(const std::vector<std::unique_ptr<Worker>>& workers, const std::vector<std::pair<Matrix, Matrix>>& chunks)
{
    for (unsigned idx = 0; idx < chunks.size(); ++idx) {
        auto& [input, target] = chunks[idx];
        auto task = this->make_task(input, target);
        workers[idx]->set_task(task);
    }
    this->state->cv_task.notify_all();
}

void ParallelSGD::wait_all_finished() const
{
    std::unique_lock<std::mutex> lock(this->state->mutex);
    this->state->cv_finished.wait(lock, [this]() {
        return this->all_params.size() == this->num_threads;
    });
}

void ParallelSGD::update_params()
{
    Matrix new_weight = std::transform_reduce(
                            this->all_params.cbegin(),
                            this->all_params.cend(),
                            xt::zeros_like(this->get_params().weight),
                            std::plus<>(),
                            std::mem_fn(&Params::weight))
        / this->num_threads;
    double new_bias = std::transform_reduce(
                          this->all_params.cbegin(),
                          this->all_params.cend(),
                          0.0,
                          std::plus<>(),
                          std::mem_fn(&Params::bias))
        / this->num_threads;
    this->core.set_params({ std::move(new_weight), new_bias });
    this->all_params.clear();
}

double ParallelSGD::compute_cost(const Matrix& input, const Matrix& target) const
{
    return this->core.compute_cost(this->core.compute_prediction(input), target);
}

void ParallelSGD::stop_workers(const std::vector<std::unique_ptr<Worker>>& workers) const
{
    for (auto& worker : workers) {
        worker->set_finished(true);
    }
    this->state->cv_task.notify_all();
    for (auto& worker : workers) {
        worker->stop();
    }
}

std::vector<double> ParallelSGD::optimize(const Matrix& input, const Matrix& target)
{
    unsigned num_epochs = this->num_epochs / this->num_step_epochs;
    std::vector<double> costs(num_epochs);

    auto workers = this->create_workers();
    auto chunks = this->make_data_chunks(input, target);

    auto engine = std::default_random_engine();
    for (unsigned epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(chunks.begin(), chunks.end(), engine);

        this->distribute_tasks(workers, chunks);
        this->wait_all_finished();
        this->update_params();
        costs[epoch] = this->compute_cost(input, target);
    }
    this->stop_workers(workers);
    return costs;
}

}
