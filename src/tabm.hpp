#pragma once
#include "value.hpp"
#include <vector>
#include <random>

class BatchEnsembleLayer {
public:
    BatchEnsembleLayer(int in_features, int out_features, int k, std::mt19937& rng);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
    int K() const { return k_; }

private:
    int in_, out_, k_;
    std::vector<std::vector<ValuePtr>> W_;
    std::vector<std::vector<ValuePtr>> r_;
    std::vector<std::vector<ValuePtr>> s_;
    std::vector<std::vector<ValuePtr>> b_;
};

class TabM {
public:
    TabM(const std::vector<int>& widths, int k, std::mt19937& rng);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
    int K() const { return k_; }

private:
    struct SharedLinear {
        int in, out;
        std::vector<std::vector<ValuePtr>> W;
        std::vector<ValuePtr> b;
        bool use_gelu;
    };

    int k_;
    BatchEnsembleLayer be_;
    std::vector<SharedLinear> shared_;
};
