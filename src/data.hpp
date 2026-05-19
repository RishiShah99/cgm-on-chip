#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Dataset {
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> X_test;
    std::vector<int> y_train;
    std::vector<int> y_test;
    std::vector<double> feat_mean;
    std::vector<double> feat_std;
    std::vector<std::string> feat_names;
};

Dataset load_ctg(const std::string& csv_path, double test_frac, uint32_t seed);
