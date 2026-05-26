// Bit-identity test for src/osdn_inference.h.
//
// RefNet below is the host-autograd "reference" mirror of the trainer's
// Net (in src/cgm_train.cpp): 7-channel input embedding, multi-layer
// OSDN stack, paper-§4.2 d-update via OSDNLayer (shared with the
// trainer), cumulative-mean-per-step pool, separate regression head
// (consumed but unused by the binary forward — same as the inference
// kernel). If you change the trainer's Net, mirror it here AND in
// src/osdn_inference.h. All three must stay in sync.
//
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
    int D_in;
    int n_layers;
    std::vector<std::vector<ValuePtr>> embed_w;  // [D_in][H]
    std::vector<ValuePtr> embed_b;
    std::vector<std::unique_ptr<OSDNLayer>> osdn_stack;
    std::vector<ValuePtr> head_w;
    ValuePtr head_b;
    std::vector<ValuePtr> pred_w;
    ValuePtr pred_b;

    RefNet(int H_, int K_, int D_in_, int n_layers_, std::mt19937& rng)
        : H(H_), D_in(D_in_), n_layers(n_layers_)
    {
        osdn_stack.reserve(n_layers_);
        for (int i = 0; i < n_layers_; ++i) {
            osdn_stack.push_back(std::make_unique<OSDNLayer>(H_, K_, rng));
        }
        std::normal_distribution<double> n01(0.0, 1.0);
        embed_w.assign(D_in_, std::vector<ValuePtr>(H_));
        embed_b.assign(H_, ValuePtr{});
        head_w.assign(H_, ValuePtr{});
        pred_w.assign(H_, ValuePtr{});
        double s  = 1.0 / std::sqrt(static_cast<double>(H_) * static_cast<double>(D_in_));
        double sH = 1.0 / std::sqrt(static_cast<double>(H_));
        for (int c = 0; c < D_in_; ++c)
            for (int h = 0; h < H_; ++h) embed_w[c][h] = v(s * n01(rng));
        for (int h = 0; h < H_; ++h) {
            embed_b[h] = v(0.0);
            head_w[h]  = v(sH * n01(rng));
            pred_w[h]  = v(sH * n01(rng));
        }
        head_b = v(0.0);
        pred_b = v(0.0);
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& row : embed_w) for (auto& q : row) p.push_back(q);
        for (auto& q : embed_b) p.push_back(q);
        for (auto& layer : osdn_stack)
            for (auto& q : layer->parameters()) p.push_back(q);
        for (auto& q : head_w) p.push_back(q);
        p.push_back(head_b);
        for (auto& q : pred_w) p.push_back(q);
        p.push_back(pred_b);
        return p;
    }

    // feat: row-major [L * D_in], same layout as osdn_inf::forward.
    double forward(const std::vector<double>& feat, int L) {
        std::vector<std::vector<ValuePtr>> x(L, std::vector<ValuePtr>(H));
        for (int t = 0; t < L; ++t) {
            for (int h = 0; h < H; ++h) {
                ValuePtr acc = embed_b[h];
                for (int c = 0; c < D_in; ++c)
                    acc = acc + embed_w[c][h] * feat[t * D_in + c];
                x[t][h] = vtanh(acc);
            }
        }
        std::vector<std::vector<ValuePtr>> y = osdn_stack[0]->forward(x);
        for (size_t i = 1; i < osdn_stack.size(); ++i) y = osdn_stack[i]->forward(y);

        std::vector<ValuePtr> pool(H, v(0.0));
        for (int t = 0; t < L; ++t)
            for (int h = 0; h < H; ++h) pool[h] = pool[h] + y[t][h];
        double inv_L = 1.0 / static_cast<double>(L);
        ValuePtr logit = head_b;
        for (int h = 0; h < H; ++h) logit = logit + head_w[h] * (pool[h] * inv_L);
        return logit->data;
    }
};

int main(int argc, char** argv) {
    int H = 16, K = 8, L = 144, D_in = 7, n_layers = 1;
    uint32_t seed = 1337;
    double tol = 1e-4;
    std::string load_weights;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto v_str = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--H")            H = std::stoi(v_str());
        else if (k == "--K")            K = std::stoi(v_str());
        else if (k == "--L")            L = std::stoi(v_str());
        else if (k == "--D-in")         D_in = std::stoi(v_str());
        else if (k == "--n-layers")     n_layers = std::stoi(v_str());
        else if (k == "--seed")         seed = static_cast<uint32_t>(std::stoul(v_str()));
        else if (k == "--tol")          tol  = std::stod(v_str());
        else if (k == "--load-weights") load_weights = v_str();
        else { std::cerr << "unknown arg: " << k << "\n"; return 2; }
    }

    std::cout << "osdn inference bit-identity test\n"
              << "  H=" << H << "  K=" << K << "  D_in=" << D_in
              << "  n_layers=" << n_layers
              << "  L=" << L << "  seed=" << seed << "  tol=" << tol << "\n";

    std::mt19937 rng(seed);
    RefNet net(H, K, D_in, n_layers, rng);
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
        std::streamsize expected_bytes =
            static_cast<std::streamsize>(params.size() * sizeof(float));
        if (bytes != expected_bytes) {
            std::cerr << "FAIL: weights file is " << bytes
                      << " bytes, model expects " << expected_bytes
                      << " (" << params.size() << " floats)\n";
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

    std::size_t expected = osdn_inf::expected_blob_floats(H, K, D_in, n_layers);
    std::cout << "  params=" << params.size() << "  expected=" << expected << "\n";
    if (params.size() != expected) {
        std::cerr << "FAIL: param count mismatch (RefNet built "
                  << params.size() << ", kernel expects " << expected << ")\n";
        return 1;
    }

    osdn_inf::Weights w;
    if (!osdn_inf::load_blob(w, H, K, D_in, n_layers, blob.data(), blob.size())) {
        std::cerr << "FAIL: load_blob rejected\n";
        return 1;
    }

    // Synthetic D_in-channel features. Channel 0 mimics glucose magnitude,
    // others are zero-mean gaussian — the exact distribution doesn't matter
    // for bit-identity testing.
    std::normal_distribution<double> n01(0.0, 1.0);
    std::vector<double> feat_d(static_cast<size_t>(L) * D_in);
    std::vector<float>  feat_f(static_cast<size_t>(L) * D_in);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < D_in; ++c) {
            double val = (c == 0) ? (100.0 + 50.0 * n01(rng)) : n01(rng);
            feat_d[t * D_in + c] = val;
            feat_f[t * D_in + c] = static_cast<float>(val);
        }
    }

    double ref_logit = net.forward(feat_d, L);

    osdn_inf::State st;
    float inf_logit = osdn_inf::forward(w, st, feat_f.data(), L);

    double diff = std::fabs(ref_logit - static_cast<double>(inf_logit));
    std::cout << std::setprecision(9);
    std::cout << "  ref_logit = " << ref_logit << "\n";
    std::cout << "  inf_logit = " << inf_logit << "\n";
    std::cout << "  |diff|    = " << diff << "\n";

    if (!std::isfinite(ref_logit) || !std::isfinite(inf_logit)) {
        std::cerr << "FAIL: non-finite logit "
                  << "(ref=" << ref_logit << ", inf=" << inf_logit << ")\n";
        return 1;
    }
    if (diff > tol) {
        std::cerr << "FAIL: diff " << diff << " > tol " << tol << "\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
