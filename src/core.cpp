#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "gradient_descent/core.hpp"
#include "gradient_descent/utils.hpp"

LinearRegressionCore::LinearRegressionCore(unsigned num_epochs, double learning_rate, double weight_decay)
    : num_epochs(num_epochs)
    , learning_rate(learning_rate)
    , weight_decay(weight_decay)
{
}

const Params& LinearRegressionCore::get_params() const
{
    return this->params;
}

void LinearRegressionCore::set_params(const Params& params)
{
    this->params = params;
}

void LinearRegressionCore::init_params(const Matrix& input, const Matrix& target)
{
    int num_features = input.shape(1);
    this->params.weight = xt::random::randn({ num_features, 1 }, 0.0, 1.0);
    this->params.bias = xt::mean(target)();
}

Matrix LinearRegressionCore::compute_prediction(const Matrix& input) const
{
    return xt::linalg::dot(input, this->params.weight) + params.bias;
}

double LinearRegressionCore::compute_cost(const Matrix& prediction, const Matrix& target) const
{
    auto cost_term = self_dot(prediction - target);
    auto reg_term = self_dot(this->params.weight);

    int num_samples = target.shape(0);
    return 1 / (2 * num_samples) * (cost_term + reg_term);
}

Params LinearRegressionCore::compute_grads(const Matrix& input, const Matrix& prediction, const Matrix& target) const
{
    int num_samples = target.shape(0);
    auto error = prediction - target;

    auto dcost_dweight = xt::linalg::dot(xt::transpose(input), error);
    auto dreg_dweight = this->weight_decay * this->params.weight;

    auto dweight = 1 / num_samples * (dcost_dweight + dreg_dweight);
    auto dbias = xt::mean(error, { 0 })();
    return { dweight, dbias };
}

void LinearRegressionCore::update_params(const Params& grads)
{
    this->params.weight -= this->learning_rate * grads.weight;
    this->params.bias -= this->learning_rate * grads.bias;
}

double LinearRegressionCore::optimize_step(const Matrix& input, const Matrix& target)
{
    auto prediction = this->compute_prediction(input);
    auto cost = this->compute_cost(prediction, target);

    auto grads = this->compute_grads(input, prediction, target);
    this->update_params(grads);
    return cost;
}

std::vector<double> LinearRegressionCore::optimize(const Matrix& input, const Matrix& target)
{
    std::vector<double> costs(num_epochs);
    for (unsigned epoch = 0; epoch < num_epochs; ++epoch) {
        costs[epoch] = this->optimize_step(input, target);
    }
    return costs;
}
