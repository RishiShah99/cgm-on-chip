#include "value.hpp"
#include "s4d.hpp"
#include "osdn.hpp"
#include "cgm_data.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <string>

namespace c {
    const std::string reset  = "\033[0m";
    const std::string bold   = "\033[1m";
    const std::string dim    = "\033[2m";
    const std::string red    = "\033[38;5;203m";
    const std::string green  = "\033[38;5;78m";
    const std::string amber  = "\033[38;5;221m";
    const std::string blue   = "\033[38;5;75m";
    const std::string cyan   = "\033[38;5;87m";
    const std::string gray   = "\033[38;5;243m";
    const std::string white  = "\033[38;5;255m";
    const std::string teal   = "\033[38;5;80m";
    const std::string mag    = "\033[38;5;177m";
}

struct Args {
    std::string arch = "both";
    std::string csv = "data/ohio_t1dm.csv";
    std::string s4d_weights = "results/s4d-ohio.weights.bin";
    std::string osdn_weights = "results/osdn-ohio.weights.bin";
    int H = 16, N = 8;
    int lookback = 144, horizon = 12, step_min = 5;
    int post_bolus = 240;
    double hypo = 70.0;
    double dt = 0.05;
    int window_stride = 12;
    int n_show = 4;
    int score_max = 200;
    int osdn_layers = 1;
    uint32_t seed = 42;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--arch") a.arch = next();
        else if (k == "--csv") a.csv = next();
        else if (k == "--load-weights-s4d") a.s4d_weights = next();
        else if (k == "--load-weights-osdn") a.osdn_weights = next();
        else if (k == "--H") a.H = std::stoi(next());
        else if (k == "--N") a.N = std::stoi(next());
        else if (k == "--lookback") a.lookback = std::stoi(next());
        else if (k == "--horizon") a.horizon = std::stoi(next());
        else if (k == "--step-min") a.step_min = std::stoi(next());
        else if (k == "--post-bolus") a.post_bolus = std::stoi(next());
        else if (k == "--hypo") a.hypo = std::stod(next());
        else if (k == "--dt") a.dt = std::stod(next());
        else if (k == "--window-stride") a.window_stride = std::stoi(next());
        else if (k == "--n-show") a.n_show = std::stoi(next());
        else if (k == "--score-max") a.score_max = std::stoi(next());
        else if (k == "--osdn-layers") a.osdn_layers = std::stoi(next());
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next()));
        else { std::cerr << "unknown arg: " << k << "\n"; std::exit(2); }
    }
    return a;
}

struct Net {
    int H;
    std::string arch;
    std::vector<ValuePtr> embed_w, embed_b;
    std::unique_ptr<S4DLayer>  s4d;
    std::vector<std::unique_ptr<OSDNLayer>> osdn_stack;
    std::vector<ValuePtr> head_w;
    ValuePtr head_b;

    Net(const std::string& arch_, int H_, int N_, double dt_, int n_osdn_layers, std::mt19937& rng)
        : H(H_), arch(arch_)
    {
        if (arch_ == "s4d") {
            s4d = std::make_unique<S4DLayer>(H_, N_, dt_, rng);
        } else {
            int nl = n_osdn_layers > 0 ? n_osdn_layers : 1;
            osdn_stack.reserve(nl);
            for (int i = 0; i < nl; ++i)
                osdn_stack.push_back(std::make_unique<OSDNLayer>(H_, N_, rng));
        }
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
        if (arch == "s4d") {
            for (auto& q : s4d->parameters()) p.push_back(q);
        } else {
            for (auto& layer : osdn_stack)
                for (auto& q : layer->parameters()) p.push_back(q);
        }
        for (auto& q : head_w) p.push_back(q);
        p.push_back(head_b);
        return p;
    }

    ValuePtr forward(const std::vector<double>& sig) {
        int L = static_cast<int>(sig.size());
        std::vector<std::vector<ValuePtr>> x(L, std::vector<ValuePtr>(H));
        for (int t = 0; t < L; ++t) {
            ValuePtr xv = v(sig[t]);
            for (int h = 0; h < H; ++h) x[t][h] = vtanh(embed_w[h] * xv + embed_b[h]);
        }
        std::vector<std::vector<ValuePtr>> y;
        if (arch == "s4d") {
            y = s4d->forward(x);
        } else {
            y = osdn_stack[0]->forward(x);
            for (size_t i = 1; i < osdn_stack.size(); ++i) y = osdn_stack[i]->forward(y);
        }
        std::vector<ValuePtr> pool(H, v(0.0));
        for (int t = 0; t < L; ++t) for (int h = 0; h < H; ++h) pool[h] = pool[h] + y[t][h];
        double inv_L = 1.0 / static_cast<double>(L);
        ValuePtr logit = head_b;
        for (int h = 0; h < H; ++h) logit = logit + head_w[h] * (pool[h] * inv_L);
        return logit;
    }
};

static bool load_weights(Net& net, const std::string& path) {
    std::ifstream wf(path, std::ios::binary);
    if (!wf.is_open()) {
        std::cerr << "  WARN: cannot open " << path << "\n";
        return false;
    }
    auto params = net.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        float fv = 0.0f;
        wf.read(reinterpret_cast<char*>(&fv), sizeof(float));
        if (!wf) {
            std::cerr << "  WARN: weight file " << path << " truncated at param " << i
                      << "/" << params.size() << "\n";
            return false;
        }
        params[i]->data = static_cast<double>(fv);
    }
    return true;
}

static double sig(double z) { return 1.0 / (1.0 + std::exp(-z)); }

static void plot_cgm(const std::vector<double>& bg, double hypo_thr) {
    const int W = 72, Hh = 8;
    const double y_min = 40.0, y_max = 280.0;
    std::vector<int> col_top(W, -1), col_bot(W, -1);
    int N_pts = static_cast<int>(bg.size());
    for (int i = 0; i < N_pts; ++i) {
        int c = i * W / N_pts;
        if (c >= W) c = W - 1;
        double y = bg[i];
        int row = static_cast<int>((y_max - y) / (y_max - y_min) * (Hh * 2));
        row = std::clamp(row, 0, Hh * 2 - 1);
        if (col_top[c] < 0 || row < col_top[c]) col_top[c] = row;
        if (col_bot[c] < 0 || row > col_bot[c]) col_bot[c] = row;
    }
    int hypo_row = static_cast<int>((y_max - hypo_thr) / (y_max - y_min) * (Hh * 2));
    int hyper_row = static_cast<int>((y_max - 180.0) / (y_max - y_min) * (Hh * 2));

    for (int row = 0; row < Hh; ++row) {
        std::cout << "  ";
        int rr1 = row * 2, rr2 = row * 2 + 1;
        double y_label = y_max - (rr1 + 0.5) / (Hh * 2) * (y_max - y_min);
        std::cout << c::gray << std::setw(3) << static_cast<int>(y_label) << " │" << c::reset;

        for (int x = 0; x < W; ++x) {
            int rt = col_top[x], rb = col_bot[x];
            bool hit_top = (rt >= rr1 && rt <= rr2);
            bool hit_bot = (rb >= rr1 && rb <= rr2);
            bool hypo_band = (rr1 <= hypo_row && hypo_row <= rr2);
            bool hyper_band = (rr1 <= hyper_row && hyper_row <= rr2);

            double y_at = y_max - ((rr1 + rr2) * 0.5) / (Hh * 2) * (y_max - y_min);
            std::string color = c::cyan;
            if (y_at < hypo_thr)      color = c::red;
            else if (y_at > 180.0)    color = c::amber;

            if (hit_top || hit_bot)         std::cout << color << "●" << c::reset;
            else if (hypo_band || hyper_band) std::cout << c::gray << "·" << c::reset;
            else                              std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout << "      " << c::gray;
    for (int x = 0; x < W; ++x) std::cout << "─";
    std::cout << c::reset << "\n";
    std::cout << "      " << c::gray << "-12h" << std::string(W - 7, ' ') << "now" << c::reset << "\n";
}

static void plot_prob_row(const std::string& label, const std::string& color_tag,
                          double prob, int true_label, double inf_ms)
{
    const int bar_w = 36;
    int filled = static_cast<int>(prob * bar_w);
    std::string col = prob >= 0.5 ? c::red : c::green;
    std::cout << "  " << color_tag << std::setw(5) << std::left << label << c::reset << " ";
    for (int i = 0; i < bar_w; ++i) std::cout << (i < filled ? col + "█" : c::gray + "·");
    std::cout << c::reset << "  " << col << std::fixed << std::setprecision(3) << prob << c::reset;
    bool right = (prob >= 0.5) == (true_label == 1);
    std::string verdict_col = right ? c::green : c::red;
    std::string mark = right ? "✓" : "✗";
    std::cout << "  " << verdict_col << mark << c::reset
              << c::gray << "   " << std::setprecision(1) << inf_ms << " ms" << c::reset << "\n";
}

struct ScoredWindow {
    int idx;
    double s4d_logit;
    double osdn_logit;
};

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    std::cout << "\n" << c::bold << c::white
              << "  CGM-DEMO  ·  pure-C++ hypoglycemia prediction on real Ohio T1DM"
              << c::reset << "\n";
    std::cout << "  " << c::gray
              << "============================================================"
              << c::reset << "\n";

    auto records = load_cgm_csv(a.csv);
    // Step 2 (patient-disjoint split): hardcode Ohio T1DM 8/2/2 split here.
    // Step 5 will rewrite this demo around the strict header-validating loader.
    PatientSplitPolicy policy;
    policy.val_ids  = {"584", "588"};
    policy.test_ids = {"591", "596"};
    auto ds = make_windows(records, a.lookback, a.horizon, a.step_min,
                           a.hypo, a.post_bolus, policy, a.seed, a.window_stride);
    std::vector<std::vector<double>> raw_lookback;
    raw_lookback.reserve(ds.test.size());
    for (const auto& w : ds.test) raw_lookback.push_back(w.lookback);
    double mean = 0.0, std_ = 1.0;
    normalize_inplace(ds, mean, std_);
    std::cout << "  " << c::gray << "loaded " << records.size() << " patients · "
              << ds.test.size() << " test windows · z=(" << std::fixed << std::setprecision(2)
              << mean << "," << std_ << ")" << c::reset << "\n";

    bool want_s4d  = (a.arch == "s4d"  || a.arch == "both");
    bool want_osdn = (a.arch == "osdn" || a.arch == "both");

    std::mt19937 rng_s(123);
    std::mt19937 rng_o(456);
    std::unique_ptr<Net> s4d_net, osdn_net;
    if (want_s4d) {
        s4d_net = std::make_unique<Net>("s4d", a.H, a.N, a.dt, 1, rng_s);
        if (!load_weights(*s4d_net, a.s4d_weights)) want_s4d = false;
        else std::cout << "  " << c::blue << "S4D" << c::reset << c::gray
                       << "  loaded " << s4d_net->parameters().size()
                       << " params from " << a.s4d_weights << c::reset << "\n";
    }
    if (want_osdn) {
        osdn_net = std::make_unique<Net>("osdn", a.H, a.N, a.dt, a.osdn_layers, rng_o);
        if (!load_weights(*osdn_net, a.osdn_weights)) want_osdn = false;
        else std::cout << "  " << c::mag << "OSDN" << c::reset << c::gray
                       << " loaded " << osdn_net->parameters().size()
                       << " params from " << a.osdn_weights
                       << " (" << a.osdn_layers << " layer" << (a.osdn_layers > 1 ? "s)" : ")")
                       << c::reset << "\n";
    }
    if (!want_s4d && !want_osdn) {
        std::cerr << "  no models loaded — abort\n";
        return 1;
    }

    int n_score = std::min<int>(a.score_max, static_cast<int>(ds.test.size()));
    std::cout << "  " << c::gray << "scoring " << n_score << " of " << ds.test.size()
              << " test windows to curate examples..." << c::reset << "\n";
    std::vector<ScoredWindow> scored;
    scored.reserve(n_score);
    for (int i = 0; i < n_score; ++i) {
        ScoredWindow s;
        s.idx = i;
        s.s4d_logit  = want_s4d  ? s4d_net->forward(ds.test[i].lookback)->data  : 0.0;
        s.osdn_logit = want_osdn ? osdn_net->forward(ds.test[i].lookback)->data : 0.0;
        scored.push_back(s);
    }

    auto pick = [&](bool want_hypo, bool want_both_agree) -> int {
        int best_idx = -1;
        double best_score = want_hypo ? -1e9 : 1e9;
        for (const auto& s : scored) {
            int lab = ds.test[s.idx].label;
            if (want_hypo && lab != 1) continue;
            if (!want_hypo && lab != 0) continue;
            double both = want_s4d && want_osdn
                ? std::min(sig(s.s4d_logit), sig(s.osdn_logit))
                : sig(want_s4d ? s.s4d_logit : s.osdn_logit);
            double avg = want_s4d && want_osdn
                ? 0.5 * (sig(s.s4d_logit) + sig(s.osdn_logit))
                : sig(want_s4d ? s.s4d_logit : s.osdn_logit);
            double score = want_both_agree ? both : avg;
            if (want_hypo  && score > best_score) { best_score = score; best_idx = s.idx; }
            if (!want_hypo && score < best_score) { best_score = score; best_idx = s.idx; }
        }
        return best_idx;
    };

    std::vector<int> show_idx;
    int idx_hypo = pick(true,  true);   if (idx_hypo  >= 0) show_idx.push_back(idx_hypo);
    int idx_safe = pick(false, true);   if (idx_safe  >= 0) show_idx.push_back(idx_safe);
    int idx_post_bolus = -1;
    for (const auto& s : scored) {
        if (ds.test[s.idx].label == 1 && ds.test[s.idx].in_post_bolus_window) {
            idx_post_bolus = s.idx; break;
        }
    }
    if (idx_post_bolus >= 0) show_idx.push_back(idx_post_bolus);
    if ((int)show_idx.size() < a.n_show) {
        for (int j = 0; j < (int)scored.size() && (int)show_idx.size() < a.n_show; ++j) {
            if (std::find(show_idx.begin(), show_idx.end(), j) == show_idx.end())
                show_idx.push_back(j);
        }
    }
    if ((int)show_idx.size() > a.n_show) show_idx.resize(a.n_show);

    const char* labels[] = {"clear hypo (both models agree)",
                            "clear safe (both models agree)",
                            "post-bolus hypo (the hard case)",
                            "additional window"};

    for (size_t k = 0; k < show_idx.size(); ++k) {
        int i = show_idx[k];
        const auto& w = ds.test[i];
        std::cout << "\n" << c::bold << c::white
                  << "  window #" << (k + 1) << "  ·  " << labels[std::min(k, sizeof(labels)/sizeof(*labels) - 1)]
                  << c::reset << "\n";
        std::cout << "  " << c::gray << "patient " << w.patient_id
                  << "   t_anchor=" << w.t_anchor_min << " min"
                  << "   future_min_glucose=" << std::fixed << std::setprecision(1) << w.future_min_glucose
                  << " mg/dL   true=" << (w.label == 1 ? "hypo" : "safe")
                  << (w.in_post_bolus_window ? "   [post-bolus window]" : "")
                  << c::reset << "\n\n";

        std::cout << "  " << c::gray << "12-hour CGM trace (mg/dL):" << c::reset << "\n";
        plot_cgm(raw_lookback[i], a.hypo);
        std::cout << "\n";

        if (want_s4d) {
            auto t0 = std::chrono::steady_clock::now();
            double z = s4d_net->forward(w.lookback)->data;
            auto t1 = std::chrono::steady_clock::now();
            double ms = 1000.0 * std::chrono::duration<double>(t1 - t0).count();
            plot_prob_row("S4D", c::blue, sig(z), w.label, ms);
        }
        if (want_osdn) {
            auto t0 = std::chrono::steady_clock::now();
            double z = osdn_net->forward(w.lookback)->data;
            auto t1 = std::chrono::steady_clock::now();
            double ms = 1000.0 * std::chrono::duration<double>(t1 - t0).count();
            plot_prob_row("OSDN", c::mag, sig(z), w.label, ms);
        }
        std::cout << "  " << c::gray
                  << "------------------------------------------------------------"
                  << c::reset << "\n";
    }

    std::cout << "\n  " << c::bold << c::white
              << "summary" << c::reset << c::gray
              << "   S4D and OSDN both train end-to-end via scalar autograd in pure C++ stdlib."
              << c::reset << "\n";
    std::cout << "  " << c::gray
              << "        Same wrapper, same loss, same data — only the recurrent layer differs."
              << c::reset << "\n\n";
    return 0;
}
