#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "gradient_descent/core.hpp"
#include "gradient_descent/utils.hpp"

namespace LinearRegression {

Core::Core(double learning_rate, double weight_decay) noexcept
    : learning_rate(learning_rate)
    , weight_decay(weight_decay)
{
}

const Params& Core::get_params() const
{
    return params;
}

void Core::set_params(const Params& params)
{
    Core::params = params;
}

void Core::init_params(const Matrix& input, const Matrix& target)
{
    int num_features = input.shape(1);
    params.weight = xt::random::randn({ num_features, 1 }, 0.0, 1.0);
    params.bias = xt::mean(target)();
}

Matrix Core::compute_prediction(const Matrix& input) const
{
    return xt::linalg::dot(input, params.weight) + params.bias;
}

double Core::compute_cost(const Matrix& prediction, const Matrix& target) const
{
    auto cost_term = self_dot(prediction - target);
    auto reg_term = weight_decay * self_dot(params.weight);

    double num_samples = target.shape(0);
    return 1 / (2 * num_samples) * (cost_term + reg_term);
}

Params Core::compute_grads(const Matrix& input, const Matrix& error) const
{
    auto dcost_dweight = xt::linalg::dot(xt::transpose(input), error);
    auto dreg_dweight = weight_decay * params.weight;

    double num_samples = input.shape(0);
    auto dweight = 1 / num_samples * (dcost_dweight + dreg_dweight);

    auto dbias = xt::mean(error)();
    return { dweight, dbias };
}

void Core::update_params(const Params& grads)
{
    params.weight -= learning_rate * grads.weight;
    params.bias -= learning_rate * grads.bias;
}

void Core::optimize_step(const Matrix& input, const Matrix& prediction, const Matrix& target)
{
    update_params(compute_grads(input, prediction - target));
}

}
