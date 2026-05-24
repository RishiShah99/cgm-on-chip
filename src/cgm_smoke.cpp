#include "value.hpp"
#include "s4d.hpp"
#include "osdn.hpp"
#include "optim.hpp"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <string>
#include <memory>

static constexpr double PI = 3.14159265358979323846;
static constexpr double GLUC_MEAN = 110.0;
static constexpr double GLUC_STD  = 35.0;

struct CGMSample {
    std::vector<double> signal;
    int label;
};

static CGMSample make_sample(std::mt19937& rng, bool force_hypo) {
    int L = 100;
    CGMSample s;
    s.signal.resize(L);
    std::normal_distribution<double> noise(0.0, 4.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    int n_meals = 1 + static_cast<int>(u01(rng) * 2);
    std::vector<int> meal_starts;
    for (int m = 0; m < n_meals; ++m)
        meal_starts.push_back(static_cast<int>(u01(rng) * (L - 30)));
    for (int t = 0; t < L; ++t) {
        double bg = GLUC_MEAN;
        for (int ms : meal_starts) {
            int dt = t - ms;
            if (dt >= 0 && dt < 30) {
                double phase = PI * static_cast<double>(dt) / 30.0;
                bg += 50.0 * std::sin(phase);
            }
        }
        bg += noise(rng);
        s.signal[t] = bg;
    }
    if (force_hypo) {
        int hypo_start = 60 + static_cast<int>(u01(rng) * 30);
        int hypo_len   = 10 + static_cast<int>(u01(rng) * 10);
        for (int t = hypo_start; t < std::min(L, hypo_start + hypo_len); ++t)
            s.signal[t] = 55.0 + noise(rng);
        s.label = 1;
    } else {
        s.label = 0;
    }
    return s;
}

static double sigmoid_d(double z) { return 1.0 / (1.0 + std::exp(-z)); }

int main(int argc, char** argv) {
    std::string arch = "s4d";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--arch" && i + 1 < argc) arch = argv[++i];
    }
    if (arch != "s4d" && arch != "osdn") {
        std::cerr << "unknown --arch: " << arch << " (s4d|osdn)\n";
        return 2;
    }

    const uint32_t seed = 7;
    const int H = 4;
    const int K = 8;
    const double dt = 0.05;
    const int n_train = 60;
    const int n_val   = 20;
    const int epochs  = 8;
    const double lr   = 0.02;
    const double wd   = 1e-5;

    std::mt19937 rng(seed);

    std::vector<CGMSample> train, val;
    for (int i = 0; i < n_train; ++i) train.push_back(make_sample(rng, i % 2 == 0));
    for (int i = 0; i < n_val;   ++i) val.push_back(  make_sample(rng, i % 2 == 0));

    std::normal_distribution<double> n01(0.0, 1.0);
    std::vector<ValuePtr> embed_w(H), embed_b(H), head_w(H);
    for (int h = 0; h < H; ++h) {
        embed_w[h] = v(0.5 * n01(rng));
        embed_b[h] = v(0.0);
        head_w[h]  = v(0.3 * n01(rng));
    }
    ValuePtr head_b = v(0.0);

    std::unique_ptr<S4DLayer>  s4d_layer;
    std::unique_ptr<OSDNLayer> osdn_layer;
    if (arch == "s4d")  s4d_layer  = std::make_unique<S4DLayer>(H, K, dt, rng);
    else                osdn_layer = std::make_unique<OSDNLayer>(H, K, rng);

    auto layer_forward = [&](const std::vector<std::vector<ValuePtr>>& x) {
        return arch == "s4d" ? s4d_layer->forward(x) : osdn_layer->forward(x);
    };

    std::vector<ValuePtr> params;
    for (auto& p : embed_w) params.push_back(p);
    for (auto& p : embed_b) params.push_back(p);
    auto lp = arch == "s4d" ? s4d_layer->parameters() : osdn_layer->parameters();
    for (auto& p : lp) params.push_back(p);
    for (auto& p : head_w) params.push_back(p);
    params.push_back(head_b);

    Adam optim(params, lr, 0.9, 0.999, 1e-8, wd);

    auto forward_one = [&](const CGMSample& s) -> ValuePtr {
        int L = static_cast<int>(s.signal.size());
        std::vector<std::vector<ValuePtr>> x(L, std::vector<ValuePtr>(H));
        for (int t = 0; t < L; ++t) {
            double xt = (s.signal[t] - GLUC_MEAN) / GLUC_STD;
            ValuePtr xv = v(xt);
            for (int h = 0; h < H; ++h)
                x[t][h] = vtanh(embed_w[h] * xv + embed_b[h]);
        }
        auto y = layer_forward(x);
        std::vector<ValuePtr> pool(H, v(0.0));
        for (int t = 0; t < L; ++t)
            for (int h = 0; h < H; ++h)
                pool[h] = pool[h] + y[t][h];
        double inv_L = 1.0 / static_cast<double>(L);
        for (int h = 0; h < H; ++h) pool[h] = pool[h] * inv_L;
        ValuePtr logit = head_b;
        for (int h = 0; h < H; ++h) logit = logit + head_w[h] * pool[h];
        return logit;
    };

    auto bce_loss = [&](ValuePtr logit, int y) -> ValuePtr {
        ValuePtr softplus = vlog(v(1.0) + vexp(v(-1.0) * logit));
        return (1 - y) * logit + softplus;
    };

    auto evaluate = [&](const std::vector<CGMSample>& ds) -> std::pair<double, double> {
        double loss_sum = 0.0;
        int correct = 0;
        for (const auto& s : ds) {
            ValuePtr logit = forward_one(s);
            double p = sigmoid_d(logit->data);
            int pred = p >= 0.5 ? 1 : 0;
            if (pred == s.label) correct++;
            double y = static_cast<double>(s.label);
            loss_sum += (1.0 - y) * logit->data + std::log(1.0 + std::exp(-logit->data));
        }
        return { loss_sum / ds.size(), static_cast<double>(correct) / ds.size() };
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << arch << "-CGM smoke test  H=" << H << "  K=" << K
              << "  params=" << params.size()
              << "  train=" << train.size() << "  val=" << val.size() << "\n\n";

    auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);
        double loss_sum = 0.0;
        for (const auto& s : train) {
            optim.zero_grad();
            ValuePtr logit = forward_one(s);
            ValuePtr loss  = bce_loss(logit, s.label);
            loss_sum += loss->data;
            loss->backward();
            optim.step();
        }
        double avg = loss_sum / train.size();
        auto [vl, va] = evaluate(val);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "ep " << std::setw(2) << epoch
                  << "  train_loss=" << avg
                  << "  val_loss="   << vl
                  << "  val_acc="    << va
                  << "  " << std::setprecision(1) << elapsed << "s\n"
                  << std::setprecision(4);
    }
    return 0;
}
