#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename T>
void dump_csv(const std::string& path, const Eigen::MatrixBase<T>& matrix)
{
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Matrix dump failed: file can not be open");
    }
    file << matrix.format(CSVFormat);
};

Eigen::MatrixXd load_csv(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Matrix load failed: file can not be open");
    }
    unsigned num_rows = 0;
    std::vector<double> values;
    std::string line, value;

    while (std::getline(file, line)) {
        std::stringstream line_stream(line);

        while (std::getline(line_stream, value, ',')) {
            values.push_back(std::stod(value));
        }
        ++num_rows;
    }
    return Eigen::Map<Eigen::Matrix<
        double,
        Eigen::Dynamic,
        Eigen::Dynamic,
        Eigen::RowMajor>>(values.data(), num_rows, values.size() / num_rows);
}
