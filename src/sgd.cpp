#include "gradient_descent/sgd.hpp"

namespace LinearRegression {

std::vector<double> SGD::optimize(const Matrix& input, const Matrix& target)
{
    std::vector<double> costs(this->num_epochs);

    for (unsigned epoch = 0; epoch < this->num_epochs; ++epoch) {
        auto prediction = this->core.compute_prediction(input);
        costs[epoch] = this->core.compute_cost(prediction, target);
        this->core.optimize_step(input, prediction, target);
    }
    return costs;
}

}