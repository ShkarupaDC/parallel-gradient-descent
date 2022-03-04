#pragma once

#include "gradient_descent/core.hpp"
#include "gradient_descent/types.hpp"
#include "gradient_descent/utils.hpp"

namespace LinearRegression {

class Interface {
public:
    explicit Interface(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize);

    virtual ~Interface() = default;

    void fit(const Matrix& input, const Matrix& target);

    Matrix predict(const Matrix& input) const;

    const Params& get_params() const;

protected:
    virtual std::vector<double> optimize(const Matrix& input, const Matrix& target) = 0;

    unsigned num_epochs;
    bool normalize;
    Core core;
    Scaler scaler;
};

}
