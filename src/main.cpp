#include <iostream>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "gradient_descent/regression.hpp"

int main(int argc, char* argv[])
{
    auto regression = LinearRegression { 10, 0.01, 0.001, true };
    // Fit
    Matrix train_input = xt::random::randn({ 200, 10 }, 0.0, 1.0);
    Matrix train_target = xt::random::randn({ 200, 1 }, 0.0, 1.0);
    regression.fit(train_input, train_target);
    // Predict
    Matrix test_input = xt::random::randn({ 50, 10 }, 0.0, 1.0);
    Matrix result = regression.predict(test_input);
    // Print parameters
    auto params = regression.get_weight_and_bias();
    std::cout << "Weight:\n"
              << params.first << std::endl;
    std::cout << "Bias:\n"
              << params.second << std::endl;
    return 0;
}
