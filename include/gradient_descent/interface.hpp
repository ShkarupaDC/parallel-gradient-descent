#pragma once

#include "gradient_descent/core.hpp"
#include "gradient_descent/types.hpp"
#include "gradient_descent/utils.hpp"

namespace LinearRegression {

class Interface {
public:
    explicit Interface(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize) noexcept;

    virtual ~Interface() = default;

    void fit(const Matrix& input, const Matrix& target);

    Matrix predict(const Matrix& input) const;

    const Params& get_params() const;

protected:
    virtual std::vector<double> optimize(const Matrix& input, const Matrix& target) = 0;

    const unsigned num_epochs;
    const bool normalize;
    Core core;
    Scaler scaler;
};

}
