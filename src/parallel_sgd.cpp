#include "profiler.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <random>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "gradient_descent/parallel_sgd.hpp"

namespace LinearRegression {

ParallelSGD::ParallelSGD(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize, unsigned num_step_epochs, unsigned num_threads)
    : Interface(num_epochs, learning_rate, weight_decay, normalize)
    , pool(num_threads)
    , num_step_epochs(num_step_epochs)
    , num_threads(num_threads)
{
}

std::vector<ParallelSGD::Chunk> ParallelSGD::get_data_chunks(const Matrix& input, const Matrix& target) const
{
    int num_samples = input.shape(0);
    int split_size = ceil(num_samples / (float)this->num_threads);

    std::vector<Chunk> chunks;
    chunks.reserve(this->num_threads);

    for (unsigned idx = 0; idx < this->num_threads; ++idx) {
        int start = idx * split_size;
        int end = std::min(start + split_size, num_samples);

        chunks.push_back({
            xt::view(input, xt::range(start, end), xt::all()),
            xt::view(target, xt::range(start, end), xt::all()),
        });
    }
    return chunks;
}

Params ParallelSGD::task(const Matrix& input, const Matrix& target) const
{
    auto new_core = this->core;
    for (unsigned epoch = 0; epoch < this->num_step_epochs; ++epoch) {
        new_core.optimize_step(
            input, new_core.compute_prediction(input), target);
    }
    return std::move(new_core.get_params());
}

std::vector<Params> ParallelSGD::optimize_parallel(const std::vector<Chunk>& chunks)
{
    std::vector<std::future<Params>> futures;
    futures.reserve(this->num_threads);

    for (const auto& chunk : chunks) {
        futures.push_back(this->pool.add_task(&ParallelSGD::task, this, chunk.input, chunk.target));
    }

    std::vector<Params> params;
    params.reserve(this->num_threads);

    for (auto&& future : futures) {
        params.emplace_back(future.get());
    }
    return params;
}

void ParallelSGD::update_params(const std::vector<Params>& params)
{
    Matrix new_weight = std::transform_reduce(
                            params.cbegin(),
                            params.cend(),
                            xt::zeros_like(this->get_params().weight),
                            std::plus<>(),
                            std::mem_fn(&Params::weight))
        / this->num_threads;
    double new_bias = std::transform_reduce(
                          params.cbegin(),
                          params.cend(),
                          0.0,
                          std::plus<>(),
                          std::mem_fn(&Params::bias))
        / this->num_threads;
    this->core.set_params({ std::move(new_weight), new_bias });
}

std::vector<double> ParallelSGD::optimize(const Matrix& input, const Matrix& target)
{
    auto engine = std::default_random_engine();
    std::vector<double> costs(num_epochs);

    std::vector<ParallelSGD::Chunk> chunks = [this, &input, &target]() {
        // LOG_DURATION("Generate data chunks");
        return this->get_data_chunks(input, target);
    }();
    for (unsigned epoch = 0; epoch < this->num_epochs / this->num_step_epochs; ++epoch) {
        // std::cout << "Epoch: " << epoch << std::endl;
        std::shuffle(chunks.begin(), chunks.end(), engine);

        std::vector<Params> params = [this, &chunks]() {
            // LOG_DURATION("Optimize parallel");
            return this->optimize_parallel(chunks);
        }();
        {
            // LOG_DURATION("Update params");
            this->update_params(std::move(params));
        }
        {
            // LOG_DURATION("Compute cost");
            costs[epoch] = this->core.compute_cost(this->core.compute_prediction(input), target);
        }
    }
    return costs;
}

}
