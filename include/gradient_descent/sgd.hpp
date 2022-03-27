#pragma once

#include "gradient_descent/interface.hpp"

namespace LinearRegression {

class SGD : public Interface {
public:
    using Interface::Interface;

protected:
    virtual std::vector<double> optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target) override;
};

}
