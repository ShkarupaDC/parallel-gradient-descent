#pragma once

#include "gradient_descent/core.hpp"
#include "gradient_descent/early_stopping.hpp"
#include "gradient_descent/scaler.hpp"

struct Batch {
    MatrixXd input;
    VectorXd target;
};

namespace LinearRegression {

class Interface {
public:
    Interface(unsigned num_epochs, int batch_size, double learning_rate, double weight_decay, bool normalize) noexcept;

    virtual ~Interface() = default;

    std::vector<double> fit(const Ref<const MatrixXd> input, const Ref<const VectorXd> target);

    VectorXd predict(const Ref<const MatrixXd> input) const;

    const Params& get_params() const;

protected:
    virtual std::vector<double> optimize(const Ref<const MatrixXd> input, const Ref<const VectorXd> target) = 0;

    unsigned get_batch_size(const Ref<const MatrixXd> input) const;

    const unsigned num_epochs;
    const int batch_size;
    const bool normalize;
    Core core;
    EarlyStopping early_stopping;
    Scaler scaler;
};

}

std::vector<Batch> generate_batches(const MatrixXd& input, const VectorXd& target, unsigned batch_size);
