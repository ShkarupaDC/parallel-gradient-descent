#include <iostream>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

int main(int argc, char* argv[])
{
    xt::xarray<double> range = xt::arange(1, 10, 1);
    range.reshape({ 3, 3 });

    std::cout << range << std::endl;
    std::cout << "Shape: " << xt::adapt(range.shape()) << std::endl;
    return 0;
}
