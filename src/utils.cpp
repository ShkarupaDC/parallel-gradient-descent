#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

#include "gradient_descent/utils.hpp"

double self_dot(const Matrix& matrix)
{
    return xt::linalg::dot(matrix, xt::transpose(matrix))();
}