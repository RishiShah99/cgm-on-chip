#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace osdn_inf {

static constexpr int H_MAX = 32;
static constexpr int K_MAX = 16;

struct Weights {
    int H;
    int K;
    float embed_w[H_MAX];
    float embed_b[H_MAX];
    float Wk[K_MAX][H_MAX];
    float Wq[K_MAX][H_MAX];
    float Wv[K_MAX][H_MAX];
    float wb[H_MAX];
    float wb_bias;
    float Wo[H_MAX][K_MAX];
    float D[H_MAX];
    float y_bias[H_MAX];
    float ln_gamma[K_MAX];
    float ln_beta[K_MAX];
    float log_eta;
    float log_gamma;
    float log_lambda;
    float head_w[H_MAX];
    float head_b;
};

struct State {
    float S[K_MAX][K_MAX];
    float d[K_MAX];
};

inline std::size_t expected_blob_floats(int H, int K) {
    return static_cast<std::size_t>(2 * H + 3 * K * H + H + 1 + H * K + H + H + 2 * K + 3 + H + 1);
}

inline bool load_blob(Weights& w, int H, int K, const float* blob, std::size_t n_floats) {
    if (H > H_MAX || K > K_MAX) return false;
    if (n_floats != expected_blob_floats(H, K)) return false;
    w.H = H;
    w.K = K;
    std::size_t p = 0;
    for (int h = 0; h < H; ++h) w.embed_w[h] = blob[p++];
    for (int h = 0; h < H; ++h) w.embed_b[h] = blob[p++];
    for (int i = 0; i < K; ++i)
        for (int h = 0; h < H; ++h) {
            w.Wk[i][h] = blob[p++];
            w.Wq[i][h] = blob[p++];
            w.Wv[i][h] = blob[p++];
        }
    for (int h = 0; h < H; ++h) w.wb[h] = blob[p++];
    w.wb_bias = blob[p++];
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < K; ++i) w.Wo[h][i] = blob[p++];
    for (int h = 0; h < H; ++h) w.D[h] = blob[p++];
    for (int h = 0; h < H; ++h) w.y_bias[h] = blob[p++];
    for (int i = 0; i < K; ++i) w.ln_gamma[i] = blob[p++];
    for (int i = 0; i < K; ++i) w.ln_beta[i]  = blob[p++];
    w.log_eta    = blob[p++];
    w.log_gamma  = blob[p++];
    w.log_lambda = blob[p++];
    for (int h = 0; h < H; ++h) w.head_w[h] = blob[p++];
    w.head_b = blob[p++];
    return p == n_floats;
}

inline void reset(State& st, int K) {
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) st.S[i][j] = 0.0f;
        st.d[i] = 1.0f;
    }
}

inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }

inline float forward(const Weights& w, State& st, const float* sig, int L) {
    const int H = w.H;
    const int K = w.K;
    const float eta    = std::exp(w.log_eta);
    const float gamma  = sigmoidf(w.log_gamma);
    const float lambda = sigmoidf(w.log_lambda);

    reset(st, K);

    float pool[H_MAX] = {0.0f};

    float x[H_MAX];
    float k[K_MAX], q[K_MAX], vv[K_MAX], ktilde[K_MAX], Sk[K_MAX], u[K_MAX], o[K_MAX];

    for (int t = 0; t < L; ++t) {
        const float xv = sig[t];
        for (int h = 0; h < H; ++h) {
            x[h] = std::tanh(w.embed_w[h] * xv + w.embed_b[h]);
        }

        for (int i = 0; i < K; ++i) {
            float ak = 0.0f, aq = 0.0f, av = 0.0f;
            for (int h = 0; h < H; ++h) {
                ak += w.Wk[i][h] * x[h];
                aq += w.Wq[i][h] * x[h];
                av += w.Wv[i][h] * x[h];
            }
            k[i]  = ak;
            q[i]  = aq;
            vv[i] = av;
        }

        float blogit = w.wb_bias;
        for (int h = 0; h < H; ++h) blogit += w.wb[h] * x[h];
        const float beta = sigmoidf(blogit);

        for (int i = 0; i < K; ++i) ktilde[i] = st.d[i] * k[i];

        for (int i = 0; i < K; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) acc += st.S[i][j] * k[j];
            Sk[i] = acc;
        }

        for (int i = 0; i < K; ++i) u[i] = vv[i] - Sk[i];

        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                st.S[i][j] = lambda * st.S[i][j] + beta * u[i] * ktilde[j];

        for (int i = 0; i < K; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) acc += st.S[i][j] * q[j];
            o[i] = acc;
        }

        float mean_o = 0.0f;
        for (int i = 0; i < K; ++i) mean_o += o[i];
        mean_o /= static_cast<float>(K);
        float var_o = 0.0f;
        for (int i = 0; i < K; ++i) {
            float c = o[i] - mean_o;
            var_o += c * c;
        }
        var_o /= static_cast<float>(K);
        const float inv_std = 1.0f / std::sqrt(var_o + 1e-5f);
        float on[K_MAX];
        for (int i = 0; i < K; ++i) on[i] = w.ln_gamma[i] * ((o[i] - mean_o) * inv_std) + w.ln_beta[i];

        for (int h = 0; h < H; ++h) {
            float acc = w.y_bias[h] + w.D[h] * x[h];
            for (int i = 0; i < K; ++i) acc += w.Wo[h][i] * on[i];
            pool[h] += acc;
        }

        for (int i = 0; i < K; ++i) {
            float update = std::exp(-eta * std::fabs(u[i]));
            st.d[i] = (1.0f - gamma) * st.d[i] + gamma * update;
        }
    }

    const float inv_L = 1.0f / static_cast<float>(L);
    float logit = w.head_b;
    for (int h = 0; h < H; ++h) logit += w.head_w[h] * (pool[h] * inv_L);
    return logit;
}

}
