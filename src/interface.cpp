#include "profiler.hpp"

#include "gradient_descent/interface.hpp"
#include "gradient_descent/utils.hpp"

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

void Interface::fit(const Matrix& input, const Matrix& target)
{
    core.init_params(input, target);
    auto processed = normalize ? scaler.fit_transform(input) : input;
    LOG_DURATION("SGD");
    optimize(processed, target);
}

Matrix Interface::predict(const Matrix& input) const
{
    return core.compute_prediction(normalize ? scaler.transform(input) : input);
}

const Params& Interface::get_params() const
{
    return core.get_params();
}

}
