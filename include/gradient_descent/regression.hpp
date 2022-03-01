#pragma once

#include "gradient_descent/core.hpp"
#include "gradient_descent/types.hpp"
#include "gradient_descent/utils.hpp"

class LinearRegression {
public:
    explicit LinearRegression(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize);

    void fit(const Matrix& input, const Matrix& target);

    Matrix predict(const Matrix& input) const;

    const Params& get_params() const;

protected:
    virtual std::vector<double> optimize(const Matrix& input, const Matrix& target);

    LinearRegressionCore core;
    Scaler scaler;
    bool normalize;
};
