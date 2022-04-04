#pragma once

#include "gradient_descent/core.hpp"
#include "gradient_descent/scaler.hpp"

namespace LinearRegression {

class Interface {
public:
    explicit Interface(unsigned num_epochs, double learning_rate, double weight_decay, bool normalize) noexcept;

    virtual ~Interface() = default;

    std::vector<double> fit(const Ref<const MatrixXd> input, const Ref<const VectorXd> target);

    VectorXd predict(const Ref<const MatrixXd> input) const;

    const Params& get_params() const;

protected:
    virtual std::vector<double> optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target) = 0;

    const unsigned num_epochs;
    const bool normalize;
    Core core;
    Scaler scaler;
};

}
