#include <vector>

#include "gradient_descent/regression.hpp"
#include "gradient_descent/utils.hpp"

LinearRegression::LinearRegression(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize)
    : core(num_epochs, learning_rate, weight_decay)
    , normalize(normalize)
{
    if (this->normalize) {
        this->scaler = Scaler();
    }
}

void LinearRegression::fit(const Matrix& input, const Matrix& target)
{
    this->core.init_params(input, target);
    auto processed = this->normalize ? this->scaler.fit_transform(input) : input;
    this->optimize(processed, target);
}

Matrix LinearRegression::predict(const Matrix& input) const
{
    auto processed = this->normalize ? this->scaler.transform(input) : input;
    return this->core.compute_prediction(processed);
}

const Params& LinearRegression::get_params() const
{
    return this->core.get_params();
}

std::vector<double> LinearRegression::optimize(const Matrix& input, const Matrix& target)
{
    return this->core.optimize(input, target);
}
