#pragma once

#include <thread>
#include <vector>

#include "gradient_descent/interface.hpp"
#include "gradient_descent/thread_pool.hpp"

namespace LinearRegression {

class ParallelSGD : public Interface {
public:
    explicit ParallelSGD(
        unsigned num_epochs,
        double learning_rate,
        double weight_decay,
        bool normalize,
        unsigned thread_epochs = 1,
        unsigned num_threads = std::max(1u, std::thread::hardware_concurrency() - 1)) noexcept;

protected:
    virtual std::vector<double> optimize(const Matrix& input, const Matrix& target) override;

private:
    struct Chunk {
        Matrix input, target;
    };

    Params task(const Matrix& input, const Matrix& target) const;

    std::vector<Chunk> get_data_chunks(const Matrix& input, const Matrix& target) const;

    std::vector<Params> optimize_parallel(const std::vector<Chunk>& chunks);

    void update_params(const std::vector<Params>& params);

    ThreadPool pool;
    const unsigned thread_epochs;
    const unsigned num_threads;
};

}
