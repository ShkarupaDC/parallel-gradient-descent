#pragma once

#include <xtensor/xarray.hpp>

using Matrix = xt::xarray<double>;

double self_dot(const Matrix& matrix);