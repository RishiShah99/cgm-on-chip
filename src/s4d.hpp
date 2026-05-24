#pragma once
#include "value.hpp"
#include "complex_value.hpp"
#include <complex>
#include <vector>
#include <random>

class S4DLayer {
public:
    int H;
    int N;
    double dt;

    std::vector<std::vector<std::complex<double>>> A_bar;
    std::vector<std::vector<ValuePtr>> B;
    std::vector<std::vector<ComplexValue>> C;
    std::vector<ValuePtr> D;

    S4DLayer(int H_, int N_, double dt_, std::mt19937& rng);

    std::vector<std::vector<ValuePtr>> forward(
        const std::vector<std::vector<ValuePtr>>& x) const;

    std::vector<ValuePtr> parameters() const;
};
