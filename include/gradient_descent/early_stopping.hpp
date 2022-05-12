#pragma once

class EarlyStopping {
public:
    EarlyStopping(unsigned patience = 10, double threshold = 1e-6) noexcept;

    void reset();

    void update(double new_value);

    bool should_stop();

private:
    unsigned steps;
    double min_value;
    const unsigned patience;
    const double threshold;
};
