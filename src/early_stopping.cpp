#include <limits>

#include "gradient_descent/early_stopping.hpp"

const double INFINITY = std::numeric_limits<double>::max();

EarlyStopping::EarlyStopping(unsigned patience, double threshold) noexcept
    : patience(patience)
    , threshold(threshold)
{
    reset();
}

void EarlyStopping::reset()
{
    steps = 0;
    min_value = INFINITY;
}

void EarlyStopping::update(double new_value)
{
    if (min_value == INFINITY) {
        min_value = new_value;
    } else {
        if (min_value - new_value > threshold) {
            steps = 0;
            min_value = new_value;
        } else {
            steps += 1;
        }
    }
}

bool EarlyStopping::should_stop()
{
    return steps > patience;
}
