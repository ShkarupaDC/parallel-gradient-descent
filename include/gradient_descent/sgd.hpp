#pragma once

#include "gradient_descent/interface.hpp"

namespace LinearRegression {

class SGD : public Interface {
public:
    using Interface::Interface;

protected:
    std::vector<double> optimize(const Matrix& input, const Matrix& target) override;
};

}
