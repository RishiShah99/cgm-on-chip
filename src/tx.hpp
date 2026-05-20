#pragma once
#include "value.hpp"
#include <vector>
#include <random>

class LayerNorm {
public:
    LayerNorm(int dim, std::mt19937& rng);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
private:
    int dim_;
    std::vector<ValuePtr> gamma_;
    std::vector<ValuePtr> beta_;
    double eps_;
};

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int n_heads, std::mt19937& rng);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<std::vector<ValuePtr>>& X) const;
    std::vector<ValuePtr> parameters() const;
private:
    int d_, h_, dh_;
    std::vector<std::vector<ValuePtr>> Wq_, Wk_, Wv_, Wo_;
    std::vector<ValuePtr> bq_, bk_, bv_, bo_;
};

class FFN {
public:
    FFN(int d_model, int d_hidden, std::mt19937& rng);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
private:
    int d_, dh_;
    std::vector<std::vector<ValuePtr>> W1_, W2_;
    std::vector<ValuePtr> b1_, b2_;
};

class TransformerBlock {
public:
    TransformerBlock(int d_model, int n_heads, int d_ffn, std::mt19937& rng);
    std::vector<std::vector<ValuePtr>> forward(const std::vector<std::vector<ValuePtr>>& X) const;
    std::vector<ValuePtr> parameters() const;
private:
    LayerNorm ln1_;
    MultiHeadAttention attn_;
    LayerNorm ln2_;
    FFN ffn_;
};

class FTTransformer {
public:
    FTTransformer(int n_features, int n_classes, int d_model,
                  int n_heads, int d_ffn, int n_blocks, std::mt19937& rng);
    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) const;
    std::vector<ValuePtr> parameters() const;
private:
    int n_features_, d_model_, n_classes_;
    std::vector<std::vector<ValuePtr>> feat_W_;
    std::vector<std::vector<ValuePtr>> feat_b_;
    std::vector<ValuePtr> cls_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm final_ln_;
    std::vector<std::vector<ValuePtr>> head_W_;
    std::vector<ValuePtr> head_b_;
};
