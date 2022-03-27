#include <random>

#include "gradient_descent/core.hpp"

double sample_gaussian(double dummy)
{
    static std::default_random_engine engine;
    static std::normal_distribution normal(0.0, 1.0);
    return normal(engine);
}

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

void Core::init_params(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    params.weight = VectorXd::Zero(input.cols()).unaryExpr(std::ptr_fun(sample_gaussian));
    params.bias = target.mean();
}

VectorXd Core::compute_prediction(const Ref<const MatrixXd> input) const
{
    return (input * params.weight).array() + params.bias;
}

double Core::compute_cost(const Ref<const VectorXd> prediction, const Ref<const VectorXd> target) const
{
    double cost_term = (prediction - target).norm();
    double reg_term = weight_decay * params.weight.norm();

    double num_samples = target.rows();
    return 1 / (2 * num_samples) * (cost_term + reg_term);
}

Params Core::compute_grads(const Ref<const MatrixXd> input, const Ref<const VectorXd> error) const
{
    auto dcost_dweight = input.transpose() * error;
    auto dreg_dweight = weight_decay * params.weight;

    double num_samples = input.rows();
    VectorXd dweight = 1 / num_samples * (dcost_dweight + dreg_dweight);

    auto dbias = error.sum() / error.size();
    return { std::move(dweight), dbias };
}

void Core::update_params(const Params& grads)
{
    params.weight -= learning_rate * grads.weight;
    params.bias -= learning_rate * grads.bias;
}

void Core::optimize_step(const Ref<const MatrixXd> input, const Ref<const VectorXd> prediction, const Ref<const VectorXd> target)
{
    update_params(compute_grads(input, prediction - target));
}

}
