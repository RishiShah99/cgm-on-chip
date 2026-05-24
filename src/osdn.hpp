#pragma once
#include "value.hpp"
#include <vector>
#include <random>

class OSDNLayer {
public:
    int H;
    int K;

    std::vector<std::vector<ValuePtr>> Wk;
    std::vector<std::vector<ValuePtr>> Wq;
    std::vector<std::vector<ValuePtr>> Wv;
    std::vector<ValuePtr> wb;
    ValuePtr wb_bias;
    std::vector<std::vector<ValuePtr>> Wo;
    std::vector<ValuePtr> D;
    std::vector<ValuePtr> y_bias;
    std::vector<ValuePtr> ln_gamma;
    std::vector<ValuePtr> ln_beta;
    std::vector<ValuePtr> log_lambda;

    OSDNLayer(int H_, int K_, std::mt19937& rng);

    std::vector<std::vector<ValuePtr>> forward(
        const std::vector<std::vector<ValuePtr>>& x) const;

    std::vector<ValuePtr> parameters() const;
};
