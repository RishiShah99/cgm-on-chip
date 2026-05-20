#pragma once
#include "value.hpp"
#include <vector>

class Adam {
public:
    Adam(std::vector<ValuePtr> params,
         double lr,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0);

    void set_lr(double lr);
    void zero_grad();
    void step();

private:
    std::vector<ValuePtr> params_;
    std::vector<double>   m_;
    std::vector<double>   v_;
    double lr_;
    double beta1_, beta2_, eps_, weight_decay_;
    long long t_;
};
