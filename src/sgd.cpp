#include "gradient_descent/sgd.hpp"

namespace LinearRegression {

std::vector<double> SGD::optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    std::vector<double> costs(this->num_epochs);
    for (unsigned epoch = 0; epoch < this->num_epochs; ++epoch) {
        auto prediction = core.compute_prediction(input);
        costs[epoch] = core.compute_cost(prediction, target);
        core.optimize_step(input, prediction, target);
    }
    return costs;
}

}
