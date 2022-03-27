#include <float.h>

#include "gradient_descent/scaler.hpp"

MatrixXd Scaler::fit_transform(const Ref<const MatrixXd> input)
{
    mean = input.colwise().mean();
    std = (input.rowwise() - mean).array().pow(2).colwise().sum() / input.rows();
    return transform(input);
}

MatrixXd Scaler::transform(const Ref<const MatrixXd> input) const
{
    return (input.rowwise() - mean).array().rowwise() / std.array().max(DBL_EPSILON);
}
