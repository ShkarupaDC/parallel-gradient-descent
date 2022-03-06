#include <float.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

#include "gradient_descent/utils.hpp"

Matrix Scaler::fit_transform(const Matrix& input)
{
    this->std = xt::stddev(input, { 0 }, xt::keep_dims);
    this->mean = xt::mean(input, { 0 }, xt::keep_dims);
    return this->transform(input);
}

Matrix Scaler::transform(const Matrix& input) const
{
    return (input - this->mean) / xt::maximum(this->std, DBL_EPSILON);
}

double self_dot(const Matrix& matrix)
{
    return xt::linalg::dot(xt::transpose(matrix), matrix)();
}
