#pragma once

#include <Eigen/Dense>

using Eigen::MatrixBase;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;

struct Params {
    VectorXd weight;
    double bias;
};

namespace LinearRegression {

class Core {
public:
    Core(double learning_rate, double weight_decay) noexcept;

    const Params& get_params() const;

    void set_params(const Params& params);

    void init_params(const Ref<const MatrixXd> input, const Ref<const VectorXd> target);

    VectorXd compute_prediction(const Ref<const MatrixXd> input) const;

    double compute_cost(const Ref<const VectorXd> prediction, const Ref<const VectorXd> target) const;

    void optimize_step(const Ref<const MatrixXd> input, const Ref<const VectorXd> prediction, const Ref<const VectorXd> target);

private:
    Params compute_grads(const Ref<const MatrixXd> input, const Ref<const VectorXd> error) const;

    void update_params(const Params& grads);

    Params params;
    const double learning_rate;
    const double weight_decay;
};

}
