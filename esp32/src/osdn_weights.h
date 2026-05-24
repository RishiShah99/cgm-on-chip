#pragma once
#include <cstdint>

namespace osdn_weights {

static constexpr int H = 16;
static constexpr int K = 8;
static constexpr int LOOKBACK = 144;
static constexpr float MEAN_MG_DL = 159.1924f;
static constexpr float STD_MG_DL  = 57.8329f;

extern const float BLOB[];
extern const std::size_t BLOB_LEN;

}
