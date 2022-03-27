#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::RowVectorXd;

class Scaler {
public:
    MatrixXd fit_transform(const Ref<const MatrixXd> input);

    MatrixXd transform(const Ref<const MatrixXd> input) const;

private:
    RowVectorXd std;
    RowVectorXd mean;
};
