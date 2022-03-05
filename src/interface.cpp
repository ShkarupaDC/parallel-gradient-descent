#include "profiler.hpp"

#include "gradient_descent/interface.hpp"
#include "gradient_descent/utils.hpp"

namespace LinearRegression {

Interface::Interface(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize)
    : num_epochs(num_epochs)
    , normalize(normalize)
    , core(learning_rate, weight_decay)
{
    if (this->normalize) {
        this->scaler = Scaler();
    }
}

void Interface::fit(const Matrix& input, const Matrix& target)
{
    this->core.init_params(input, target);
    auto processed = this->normalize ? this->scaler.fit_transform(input) : std::move(input);
    LOG_DURATION("SGD");
    this->optimize(processed, target);
}

Matrix Interface::predict(const Matrix& input) const
{
    return this->core.compute_prediction(this->normalize ? this->scaler.transform(input) : input);
}

const Params& Interface::get_params() const
{
    return this->core.get_params();
}

}
