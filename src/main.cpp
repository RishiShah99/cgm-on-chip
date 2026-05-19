#include "data.hpp"
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "data/CTG.csv";
    auto ds = load_ctg(path, 0.2, 42);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "train rows = " << ds.X_train.size()
              << "   test rows = " << ds.X_test.size() << "\n";
    std::cout << "features   = " << ds.X_train[0].size() << "\n";

    int tr[3] = {0, 0, 0};
    int te[3] = {0, 0, 0};
    for (int y : ds.y_train) tr[y]++;
    for (int y : ds.y_test) te[y]++;
    std::cout << "train  N=" << tr[0] << "  S=" << tr[1] << "  P=" << tr[2] << "\n";
    std::cout << "test   N=" << te[0] << "  S=" << te[1] << "  P=" << te[2] << "\n";

    std::cout << "post-norm train column means (should be ~0):\n";
    for (size_t j = 0; j < ds.feat_names.size(); ++j) {
        double s = 0.0;
        for (const auto& r : ds.X_train) s += r[j];
        s /= static_cast<double>(ds.X_train.size());
        std::cout << "  " << ds.feat_names[j] << "  mu=" << s
                  << "  raw_mean=" << ds.feat_mean[j]
                  << "  raw_std=" << ds.feat_std[j] << "\n";
    }
    return 0;
}
