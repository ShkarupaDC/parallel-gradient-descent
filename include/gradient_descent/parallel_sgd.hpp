#pragma once

#include <thread>
#include <vector>

#include "gradient_descent/interface.hpp"
#include "gradient_descent/thread_pool.hpp"

namespace LinearRegression {

class ParallelSGD : public Interface {
public:
    ParallelSGD(
        unsigned num_epochs,
        int batch_size,
        double learning_rate,
        double weight_decay,
        bool normalize,
        unsigned thread_epochs = 1,
        unsigned num_threads = std::max(1u, std::thread::hardware_concurrency() - 1)) noexcept;

protected:
    virtual std::vector<double> optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target) override;

private:
    Params task(const MatrixXd& input, const VectorXd& target) const;

    std::vector<Batch> get_thread_batches(const MatrixXd& input, const VectorXd& target) const;

    std::vector<Params> optimize_parallel(const std::vector<Batch>& thread_bacthes);

    void average_params(const std::vector<Params>& params);

    ThreadPool pool;
    const unsigned thread_epochs;
    const unsigned num_threads;
};

}
