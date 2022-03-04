#pragma once

#include "gradient_descent/types.hpp"

struct Params {
    Matrix weight;
    double bias;
};

namespace LinearRegression {

class Core {
public:
    explicit Core(double learning_rate, double weight_decay);

    const Params& get_params() const;

    void set_params(const Params& params);

    void init_params(const Matrix& input, const Matrix& target);

    Matrix compute_prediction(const Matrix& input) const;

    double compute_cost(const Matrix& prediction, const Matrix& target) const;

    void optimize_step(const Matrix& input, const Matrix& prediction, const Matrix& target);

private:
    Params compute_grads(const Matrix& input, const Matrix& prediction, const Matrix& target) const;

    void update_params(const Params& grads);

    Params params;
    double learning_rate;
    double weight_decay;
};

}
