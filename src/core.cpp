#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "gradient_descent/core.hpp"
#include "gradient_descent/utils.hpp"

namespace LinearRegression {

Core::Core(double learning_rate, double weight_decay)
    : learning_rate(learning_rate)
    , weight_decay(weight_decay)
{
}

const Params& Core::get_params() const
{
    return this->params;
}

Params&& Core::get_params()
{
    return std::move(this->params);
}

void Core::set_params(const Params& params)
{
    this->params = params;
}

void Core::init_params(const Matrix& input, const Matrix& target)
{
    int num_features = input.shape(1);
    this->params.weight = xt::random::randn({ num_features, 1 }, 0.0, 1.0);
    this->params.bias = xt::mean(target)();
}

Matrix Core::compute_prediction(const Matrix& input) const
{
    return xt::linalg::dot(input, this->params.weight) + this->params.bias;
}

double Core::compute_cost(const Matrix& prediction, const Matrix& target) const
{
    auto cost_term = self_dot(prediction - target);
    auto reg_term = this->weight_decay * self_dot(this->params.weight);

    double num_samples = target.shape(0);
    return 1 / (2 * num_samples) * (cost_term + reg_term);
}

Params Core::compute_grads(const Matrix& input, const Matrix& prediction, const Matrix& target) const
{
    double num_samples = target.shape(0);
    auto error = prediction - target;

    auto dcost_dweight = xt::linalg::dot(xt::transpose(input), error);
    auto dreg_dweight = this->weight_decay * this->params.weight;

    auto dweight = 1 / num_samples * (std::move(dcost_dweight) + std::move(dreg_dweight));
    auto dbias = xt::mean(std::move(error))();
    return { std::move(dweight), dbias };
}

void Core::update_params(const Params& grads)
{
    this->params.weight -= this->learning_rate * grads.weight;
    this->params.bias -= this->learning_rate * grads.bias;
}

void Core::optimize_step(const Matrix& input, const Matrix& prediction, const Matrix& target)
{
    this->update_params(this->compute_grads(input, prediction, target));
}

}
