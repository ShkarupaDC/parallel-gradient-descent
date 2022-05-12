#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>

#include "gradient_descent/parallel_sgd.hpp"

namespace LinearRegression {

ParallelSGD::ParallelSGD(
    unsigned num_epochs,
    int batch_size,
    double learning_rate,
    double weight_decay,
    bool normalize,
    unsigned thread_epochs,
    unsigned num_threads) noexcept
    : Interface(num_epochs, batch_size, learning_rate, weight_decay, normalize)
    , pool(num_threads)
    , thread_epochs(thread_epochs)
    , num_threads(num_threads)
{
}

std::vector<Batch> ParallelSGD::get_thread_batches(const MatrixXd& input, const VectorXd& target) const
{
    unsigned split_size = ceil(input.rows() / (double)num_threads);
    return generate_batches(input, target, split_size);
}

Params ParallelSGD::task(const MatrixXd& input, const VectorXd& target) const
{
    auto thread_core = core;
    for (unsigned epoch = 0; epoch < thread_epochs; ++epoch) {
        thread_core.optimize_step(
            input, thread_core.compute_prediction(input), target);
    }
    return thread_core.get_params();
}

std::vector<Params> ParallelSGD::optimize_parallel(const std::vector<Batch>& thread_batches)
{
    std::vector<std::future<Params>> futures;
    futures.reserve(num_threads);

    for (const auto& batch : thread_batches) {
        futures.push_back(pool.add_task(
            &ParallelSGD::task,
            this,
            std::ref(batch.input),
            std::ref(batch.target)));
    }
    std::vector<Params> params;
    params.reserve(num_threads);

    for (auto&& future : futures)
        params.push_back(future.get());
    return params;
}

void ParallelSGD::average_params(const std::vector<Params>& params)
{
    VectorXd new_weight = std::transform_reduce(
                              params.cbegin(),
                              params.cend(),
                              VectorXd::Zero(get_params().weight.size()).eval(),
                              std::plus<>(),
                              std::mem_fn(&Params::weight))
                              .array()
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

std::vector<double> ParallelSGD::optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    unsigned global_epochs = num_epochs / thread_epochs;
    std::vector<double> costs;
    costs.reserve(global_epochs);

    std::cout << "Num threads: " << num_threads << std::endl;

    auto thread_batches = get_thread_batches(input, target);
    for (unsigned epoch = 0; epoch < global_epochs; ++epoch) {
        auto params = optimize_parallel(thread_batches);
        average_params(params);
        auto cost = core.compute_cost(core.compute_prediction(input), target);
        costs.push_back(cost);

        // early_stopping.update(cost);
        // if (early_stopping.should_stop()) {
        //     break;
        // }
    }
    return costs;
}
}
