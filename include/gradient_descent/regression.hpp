#pragma once

#include <xtensor/xarray.hpp>

using Matrix = xt::xarray<double>;

class LinearRegression {
public:
    explicit LinearRegression(const unsigned num_epochs, const double learning_rate, const double weight_decay, const bool normalize);

    void fit(const Matrix& input, const Matrix& target);

    Matrix predict(const Matrix& input) const;

    std::pair<const Matrix&, const double&> get_weight_and_bias() const;

private:
    Matrix normalize_input(const Matrix& input) const;

    void init_params(const Matrix& input, const Matrix& target);

    Matrix compute_prediction(const Matrix& input) const;

    double compute_cost(const Matrix& prediction, const Matrix& target) const;

    void update_params(const Matrix& input, const Matrix& prediction, const Matrix& target);

    std::vector<double> optimize(const Matrix& input, const Matrix& target);

    Matrix weight;
    double bias;

    unsigned num_epochs;
    double learning_rate;
    double weight_decay;
    bool normalize;
};