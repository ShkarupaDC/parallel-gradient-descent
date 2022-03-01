#pragma once

#include "gradient_descent/types.hpp"

class Scaler {
public:
    Matrix fit_transform(const Matrix& input);

    Matrix transform(const Matrix& input) const;

private:
    Matrix std;
    Matrix mean;
};

double self_dot(const Matrix& matrix);
