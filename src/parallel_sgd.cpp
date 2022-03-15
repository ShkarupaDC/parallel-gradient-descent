#include <algorithm>
#include <cmath>
#include <future>
#include <random>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "gradient_descent/parallel_sgd.hpp"

namespace LinearRegression {

ParallelSGD::ParallelSGD(
    unsigned num_epochs,
    double learning_rate,
    double weight_decay,
    bool normalize,
    unsigned thread_epochs,
    unsigned num_threads) noexcept
    : Interface(num_epochs, learning_rate, weight_decay, normalize)
    , pool(num_threads)
    , thread_epochs(thread_epochs)
    , num_threads(num_threads)
{
}

std::vector<ParallelSGD::Chunk> ParallelSGD::get_data_chunks(const Matrix& input, const Matrix& target) const
{
    int num_samples = input.shape(0);
    int split_size = ceil(num_samples / (float)num_threads);

    std::vector<Chunk> chunks;
    chunks.reserve(num_threads);

    for (unsigned idx = 0; idx < num_threads; ++idx) {
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
    auto thread_core = core;
    for (unsigned epoch = 0; epoch < thread_epochs; ++epoch) {
        thread_core.optimize_step(
            input, thread_core.compute_prediction(input), target);
    }
    return thread_core.get_params();
}

std::vector<Params> ParallelSGD::optimize_parallel(const std::vector<Chunk>& chunks)
{
    std::vector<std::future<Params>> futures;
    futures.reserve(num_threads);

    for (const auto& chunk : chunks) {
        futures.push_back(pool.add_task(
            &ParallelSGD::task,
            this,
            std::ref(chunk.input),
            std::ref(chunk.target)));
    }
    std::vector<Params> params;
    params.reserve(num_threads);

    for (auto&& future : futures)
        params.push_back(future.get());
    return params;
}

void ParallelSGD::update_params(const std::vector<Params>& params)
{
    Matrix new_weight = std::transform_reduce(
                            params.cbegin(),
                            params.cend(),
                            xt::zeros_like(get_params().weight),
                            std::plus<>(),
                            std::mem_fn(&Params::weight))
        / num_threads;
    double new_bias = std::transform_reduce(
                          params.cbegin(),
                          params.cend(),
                          0.0,
                          std::plus<>(),
                          std::mem_fn(&Params::bias))
        / num_threads;
    core.set_params({ std::move(new_weight), new_bias });
}

std::vector<double> ParallelSGD::optimize(const Matrix& input, const Matrix& target)
{
    auto engine = std::default_random_engine();
    std::vector<double> costs(num_epochs);

    auto chunks = get_data_chunks(input, target);
    for (unsigned epoch = 0; epoch < num_epochs / thread_epochs; ++epoch) {
        std::shuffle(chunks.begin(), chunks.end(), engine);

        auto params = optimize_parallel(chunks);
        update_params(params);
        costs[epoch] = core.compute_cost(core.compute_prediction(input), target);
    }
    return costs;
}
}
