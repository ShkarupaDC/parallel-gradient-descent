#pragma once

#include <vector>

#include "gradient_descent/types.hpp"

struct Params {
    Matrix weight;
    double bias;
};

class LinearRegressionCore {
public:
    friend class LinearRegression;

    explicit LinearRegressionCore(unsigned num_epochs, double learning_rate, double weight_decay);

    const Params& get_params() const;

    void set_params(const Params& params);

    std::vector<double> optimize(const Matrix& input, const Matrix& target);

private:
    void init_params(const Matrix& input, const Matrix& target);

    Matrix compute_prediction(const Matrix& input) const;

    double compute_cost(const Matrix& prediction, const Matrix& target) const;

    Params compute_grads(const Matrix& input, const Matrix& prediction, const Matrix& target) const;

    void update_params(const Params& grads);

    double optimize_step(const Matrix& input, const Matrix& target);

    Params params;
    unsigned num_epochs;
    double learning_rate;
    double weight_decay;
};
