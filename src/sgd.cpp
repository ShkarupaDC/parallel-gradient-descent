#include <iomanip>
#include <iostream>

#include "gradient_descent/sgd.hpp"

namespace LinearRegression {

std::vector<double> SGD::optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target)
{
    std::vector<double> costs;
    costs.reserve(num_epochs);

    for (unsigned epoch = 0; epoch < num_epochs; ++epoch) {
        auto prediction = core.compute_prediction(input);
        auto cost = core.compute_cost(prediction, target);
        costs.push_back(cost);
        core.optimize_step(input, prediction, target);

        // early_stopping.update(cost);
        // if (early_stopping.should_stop()) {
        //     break;
        // }
    }
    return costs;
}

}
