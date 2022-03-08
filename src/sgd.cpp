#include "profiler.hpp"

#include "gradient_descent/sgd.hpp"

namespace LinearRegression {

std::vector<double> SGD::optimize(const Matrix& input, const Matrix& target)
{
    std::vector<double> costs(this->num_epochs);

    for (unsigned epoch = 0; epoch < this->num_epochs; ++epoch) {
        Matrix prediction = [this, &input]() {
            // LOG_DURATION("Make prediction");
            return this->core.compute_prediction(input);
        }();
        {
            // LOG_DURATION("Compute cost")
            costs[epoch] = this->core.compute_cost(prediction, target);
        }
        {
            // LOG_DURATION("Optimize step");
            this->core.optimize_step(input, prediction, target);
        }
    }
    return costs;
}

}
