#include <algorithm>
#include <numeric>
#include <random>

#include "profiler.hpp"

#include "gradient_descent/interface.hpp"

namespace LinearRegression {

Interface::Interface(unsigned num_epochs, int batch_size, double learning_rate, double weight_decay, bool normalize) noexcept
    : num_epochs(num_epochs)
    , batch_size(batch_size)
    , normalize(normalize)
    , core(learning_rate, weight_decay)
{
    if (Interface::normalize) {
        scaler = Scaler();
    }
}

std::vector<double> Interface::fit(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    core.init_params(input, target);
    auto processed = normalize ? scaler.fit_transform(input) : input.eval();
    LOG_DURATION("Time");
    return optimize(processed, target);
}

VectorXd Interface::predict(const Ref<const MatrixXd> input) const
{
    return core.compute_prediction(normalize ? scaler.transform(input) : input.eval());
}

const Params& Interface::get_params() const
{
    return core.get_params();
}

unsigned Interface::get_batch_size(const Ref<const MatrixXd> input) const
{
    return batch_size <= 0 ? input.rows() : std::min(batch_size, (int)input.rows());
}

}

std::vector<Batch> generate_batches(const MatrixXd& input, const VectorXd& target, unsigned batch_size)
{
    unsigned num_samples = input.rows();
    unsigned num_batches = ceil(num_samples / (float)batch_size);

    static std::default_random_engine engine;
    std::vector<unsigned> idxs(num_samples);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::shuffle(idxs.begin(), idxs.end(), engine);

    std::vector<Batch> batches;
    batches.reserve(num_batches);

    auto idxs_begin = std::make_move_iterator(idxs.begin());
    for (unsigned idx = 0; idx < num_batches; ++idx) {
        int start = idx * batch_size;
        int end = std::min(start + batch_size, num_samples);

        std::vector<unsigned> batch_idxs(std::next(idxs_begin, start), std::next(idxs_begin, end));
        batches.push_back({ input(batch_idxs, Eigen::all), target(batch_idxs) });
    }
    return batches;
}
