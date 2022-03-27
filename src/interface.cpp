#include "profiler.hpp"

#include "gradient_descent/interface.hpp"

namespace LinearRegression {

Interface::Interface(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize) noexcept
    : num_epochs(num_epochs)
    , normalize(normalize)
    , core(learning_rate, weight_decay)
{
    if (Interface::normalize) {
        scaler = Scaler();
    }
}

void Interface::fit(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    core.init_params(input, target);
    auto processed = normalize ? scaler.fit_transform(input) : input.eval();
    LOG_DURATION("SGD");
    optimize(processed, target);
}

VectorXd Interface::predict(const Ref<const MatrixXd> input) const
{
    return core.compute_prediction(normalize ? scaler.transform(input) : input.eval());
}

const Params& Interface::get_params() const
{
    return core.get_params();
}

}
