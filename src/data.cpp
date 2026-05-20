#include "data.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cmath>

static const std::vector<std::string> kFeatureNames = {
    "LB", "LBE", "AC", "FM", "UC", "DL", "DS", "DP",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency"
};
static const std::string kTargetName = "NSP";

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { out.push_back(cell); cell.clear(); }
        else cell.push_back(c);
    }
    out.push_back(cell);
    return out;
}

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static bool try_parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    try {
        size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

Dataset load_ctg(const std::string& csv_path,
                 double val_frac,
                 double test_frac,
                 uint32_t seed) {
    std::ifstream file(csv_path);
    if (!file.is_open()) throw std::runtime_error("cannot open " + csv_path);

    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("empty file");
    auto header = split_csv_line(line);

    std::unordered_map<std::string, int> col;
    for (size_t i = 0; i < header.size(); ++i) col[trim(header[i])] = static_cast<int>(i);

    std::vector<int> feat_idx;
    feat_idx.reserve(kFeatureNames.size());
    for (const auto& name : kFeatureNames) {
        auto it = col.find(name);
        if (it == col.end()) throw std::runtime_error("missing column: " + name);
        feat_idx.push_back(it->second);
    }
    auto it_t = col.find(kTargetName);
    if (it_t == col.end()) throw std::runtime_error("missing column: " + kTargetName);
    int target_idx = it_t->second;

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        auto cells = split_csv_line(line);
        if (cells.size() <= static_cast<size_t>(target_idx)) continue;

        std::vector<double> row;
        row.reserve(feat_idx.size());
        bool ok = true;
        for (int idx : feat_idx) {
            double val;
            if (!try_parse_double(trim(cells[idx]), val)) { ok = false; break; }
            row.push_back(val);
        }
        if (!ok) continue;

        double tv;
        if (!try_parse_double(trim(cells[target_idx]), tv)) continue;
        int label = static_cast<int>(tv) - 1;
        if (label < 0 || label > 2) continue;

        X.push_back(std::move(row));
        y.push_back(label);
    }
    if (X.empty()) throw std::runtime_error("no usable rows parsed");

    std::mt19937 rng(seed);
    std::vector<std::vector<size_t>> by_class(3);
    for (size_t i = 0; i < y.size(); ++i) by_class[y[i]].push_back(i);

    std::vector<size_t> train_idx, val_idx, test_idx;
    for (int c = 0; c < 3; ++c) {
        std::shuffle(by_class[c].begin(), by_class[c].end(), rng);
        size_t n_test = static_cast<size_t>(test_frac * by_class[c].size());
        size_t n_val  = static_cast<size_t>(val_frac  * by_class[c].size());
        for (size_t i = 0; i < by_class[c].size(); ++i) {
            if (i < n_test)             test_idx.push_back(by_class[c][i]);
            else if (i < n_test + n_val) val_idx.push_back(by_class[c][i]);
            else                         train_idx.push_back(by_class[c][i]);
        }
    }

    Dataset ds;
    ds.feat_names = kFeatureNames;
    ds.X_train.reserve(train_idx.size()); ds.y_train.reserve(train_idx.size());
    for (size_t i : train_idx) { ds.X_train.push_back(X[i]); ds.y_train.push_back(y[i]); }
    ds.X_val.reserve(val_idx.size());     ds.y_val.reserve(val_idx.size());
    for (size_t i : val_idx)   { ds.X_val.push_back(X[i]);   ds.y_val.push_back(y[i]); }
    ds.X_test.reserve(test_idx.size());   ds.y_test.reserve(test_idx.size());
    for (size_t i : test_idx)  { ds.X_test.push_back(X[i]);  ds.y_test.push_back(y[i]); }

    size_t F = kFeatureNames.size();
    ds.feat_mean.assign(F, 0.0);
    ds.feat_std.assign(F, 0.0);
    for (const auto& row : ds.X_train)
        for (size_t j = 0; j < F; ++j) ds.feat_mean[j] += row[j];
    double n_train = static_cast<double>(ds.X_train.size());
    for (size_t j = 0; j < F; ++j) ds.feat_mean[j] /= n_train;

    for (const auto& row : ds.X_train) {
        for (size_t j = 0; j < F; ++j) {
            double d = row[j] - ds.feat_mean[j];
            ds.feat_std[j] += d * d;
        }
    }
    for (size_t j = 0; j < F; ++j) {
        ds.feat_std[j] = std::sqrt(ds.feat_std[j] / std::max(1.0, n_train - 1.0));
        if (ds.feat_std[j] < 1e-12) ds.feat_std[j] = 1.0;
    }

    auto normalize = [&](std::vector<std::vector<double>>& M) {
        for (auto& row : M)
            for (size_t j = 0; j < F; ++j)
                row[j] = (row[j] - ds.feat_mean[j]) / ds.feat_std[j];
    };
    normalize(ds.X_train);
    normalize(ds.X_val);
    normalize(ds.X_test);

    return ds;
}
