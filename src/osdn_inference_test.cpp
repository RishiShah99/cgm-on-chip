#include "value.hpp"
#include "osdn.hpp"
#include "osdn_inference.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>

struct RefNet {
    int H;
    std::vector<ValuePtr> embed_w, embed_b;
    std::unique_ptr<OSDNLayer> osdn;
    std::vector<ValuePtr> head_w;
    ValuePtr head_b;

    RefNet(int H_, int K_, std::mt19937& rng) : H(H_) {
        osdn = std::make_unique<OSDNLayer>(H_, K_, rng);
        std::normal_distribution<double> n01(0.0, 1.0);
        embed_w.resize(H_); embed_b.resize(H_); head_w.resize(H_);
        double s = 1.0 / std::sqrt(static_cast<double>(H_));
        for (int h = 0; h < H_; ++h) {
            embed_w[h] = v(s * n01(rng));
            embed_b[h] = v(0.0);
            head_w[h]  = v(s * n01(rng));
        }
        head_b = v(0.0);
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& q : embed_w) p.push_back(q);
        for (auto& q : embed_b) p.push_back(q);
        auto lp = osdn->parameters();
        for (auto& q : lp) p.push_back(q);
        for (auto& q : head_w) p.push_back(q);
        p.push_back(head_b);
        return p;
    }

    double forward(const std::vector<double>& sig) {
        int L = static_cast<int>(sig.size());
        std::vector<std::vector<ValuePtr>> x(L, std::vector<ValuePtr>(H));
        for (int t = 0; t < L; ++t) {
            ValuePtr xv = v(sig[t]);
            for (int h = 0; h < H; ++h) x[t][h] = vtanh(embed_w[h] * xv + embed_b[h]);
        }
        auto y = osdn->forward(x);
        std::vector<ValuePtr> pool(H, v(0.0));
        for (int t = 0; t < L; ++t) for (int h = 0; h < H; ++h) pool[h] = pool[h] + y[t][h];
        double inv_L = 1.0 / static_cast<double>(L);
        ValuePtr logit = head_b;
        for (int h = 0; h < H; ++h) logit = logit + head_w[h] * (pool[h] * inv_L);
        return logit->data;
    }
};

int main(int argc, char** argv) {
    int H = 16, K = 8, L = 96;
    uint32_t seed = 1337;
    double tol = 1e-4;
    std::string load_weights;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto v_str = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--H") H = std::stoi(v_str());
        else if (k == "--K") K = std::stoi(v_str());
        else if (k == "--L") L = std::stoi(v_str());
        else if (k == "--seed") seed = static_cast<uint32_t>(std::stoul(v_str()));
        else if (k == "--tol") tol = std::stod(v_str());
        else if (k == "--load-weights") load_weights = v_str();
    }

    std::cout << "osdn inference bit-identity test\n"
              << "  H=" << H << "  K=" << K << "  L=" << L << "  seed=" << seed << "  tol=" << tol << "\n";

    std::mt19937 rng(seed);
    RefNet net(H, K, rng);

    auto params = net.parameters();

    if (!load_weights.empty()) {
        std::ifstream wf(load_weights, std::ios::binary);
        if (!wf.is_open()) {
            std::cerr << "FAIL: cannot open --load-weights " << load_weights << "\n";
            return 1;
        }
        wf.seekg(0, std::ios::end);
        std::streamsize bytes = wf.tellg();
        wf.seekg(0, std::ios::beg);
        std::streamsize expected_bytes = static_cast<std::streamsize>(params.size() * sizeof(float));
        if (bytes != expected_bytes) {
            std::cerr << "FAIL: weights file is " << bytes << " bytes, model expects "
                      << expected_bytes << " (" << params.size() << " floats)\n";
            return 1;
        }
        for (size_t i = 0; i < params.size(); ++i) {
            float fv = 0.0f;
            wf.read(reinterpret_cast<char*>(&fv), sizeof(float));
            params[i]->data = static_cast<double>(fv);
        }
        std::cout << "  loaded weights from " << load_weights << " (" << bytes << " bytes)\n";
    }

    std::vector<float> blob;
    blob.reserve(params.size());
    for (auto& p : params) blob.push_back(static_cast<float>(p->data));

    std::size_t expected = osdn_inf::expected_blob_floats(H, K);
    std::cout << "  params=" << params.size() << "  expected=" << expected << "\n";
    if (params.size() != expected) {
        std::cerr << "FAIL: param count mismatch\n";
        return 1;
    }

    osdn_inf::Weights w;
    if (!osdn_inf::load_blob(w, H, K, blob.data(), blob.size())) {
        std::cerr << "FAIL: load_blob rejected\n";
        return 1;
    }

    std::normal_distribution<double> n01(0.0, 1.0);
    std::vector<double> sig_d(L);
    std::vector<float>  sig_f(L);
    for (int t = 0; t < L; ++t) {
        double s = 100.0 + 50.0 * n01(rng);
        sig_d[t] = s;
        sig_f[t] = static_cast<float>(s);
    }

    double ref_logit = net.forward(sig_d);

    osdn_inf::State st;
    float inf_logit = osdn_inf::forward(w, st, sig_f.data(), L);

    double diff = std::fabs(ref_logit - static_cast<double>(inf_logit));
    std::cout << std::setprecision(9);
    std::cout << "  ref_logit = " << ref_logit << "\n";
    std::cout << "  inf_logit = " << inf_logit << "\n";
    std::cout << "  |diff|    = " << diff << "\n";

    if (diff > tol) {
        std::cerr << "FAIL: diff " << diff << " > tol " << tol << "\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
