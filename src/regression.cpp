#include <array>
#include <float.h>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "gradient_descent/regression.hpp"
#include "gradient_descent/utils.hpp"

LinearRegression::LinearRegression(const unsigned num_epochs, const double learning_rate, const double weight_decay, const bool normalize)
    : num_epochs(num_epochs)
    , learning_rate(learning_rate)
    , weight_decay(weight_decay)
    , normalize(normalize)
{
}

void LinearRegression::fit(const Matrix& input, const Matrix& target)
{
    this->init_params(input, target);
    auto processed = this->normalize ? this->normalize_input(input) : input;
    this->optimize(processed, target);
}

Matrix LinearRegression::predict(const Matrix& input) const
{
    auto processed = this->normalize ? this->normalize_input(input) : input;
    return this->compute_prediction(processed);
}

std::pair<const Matrix&, const double&> LinearRegression::get_weight_and_bias() const
{
    return std::pair<const Matrix&, const double&>(this->weight, this->bias);
}

Matrix LinearRegression::normalize_input(const Matrix& input) const
{
    Matrix std = xt::stddev(input, { 1 }, xt::keep_dims);
    Matrix mean = xt::mean(input, { 1 }, xt::keep_dims);
    return (input - mean) / xt::maximum(std, DBL_EPSILON);
}

void LinearRegression::init_params(const Matrix& input, const Matrix& target)
{
    int num_features = input.shape(1);
    this->weight = xt::random::randn({ num_features, 1 }, 0.0, 1.0);
    this->bias = xt::mean(target)();
}

Matrix LinearRegression::compute_prediction(const Matrix& input) const
{
    return xt::linalg::dot(input, this->weight) + this->bias;
}

double LinearRegression::compute_cost(const Matrix& prediction, const Matrix& target) const
{
    auto cost_term = self_dot(prediction - target);
    auto reg_term = self_dot(this->weight);

    int num_samples = target.shape(0);
    return 1 / (2 * num_samples) * (cost_term + reg_term);
}

void LinearRegression::update_params(const Matrix& input, const Matrix& prediction, const Matrix& target)
{
    int num_samples = target.shape(0);
    auto error = prediction - target;

    auto dcost_dweight = xt::linalg::dot(xt::transpose(input), error);
    auto dreg_dweight = this->weight_decay * this->weight;

    auto dweight = 1 / num_samples * (dcost_dweight + dreg_dweight);
    auto dbias = xt::mean(error, { 0 })();

    this->weight -= this->learning_rate * dweight;
    this->bias -= this->learning_rate * dbias;
}

std::vector<double> LinearRegression::optimize(const Matrix& input, const Matrix& target)
{
    std::vector<double> costs(this->num_epochs);

    for (unsigned epoch = 0; epoch < this->num_epochs; ++epoch) {
        auto prediction = this->compute_prediction(input);
        costs[epoch] = this->compute_cost(prediction, target);
        this->update_params(input, prediction, target);
    }

    return costs;
}