#pragma once
#include <cstdint>
#include <cstddef>

namespace osdn_weights {

// Legacy compile-time shape constants kept for the existing single-channel
// firmware in esp32/src/main.cpp. Step 7 will switch the firmware to the
// runtime BLOB_* values below (which come from osdn_blob's header) so the
// constants and the blob can't disagree silently.
static constexpr int   H          = 16;
static constexpr int   K          = 8;
static constexpr int   LOOKBACK   = 144;
static constexpr float MEAN_MG_DL = 159.1924f;
static constexpr float STD_MG_DL  = 57.8329f;

// Runtime shape constants emitted by blob_to_header from the blob header.
// Firmware should prefer these over the legacy constexpr above; mismatch
// means the blob has shifted away from the values main.cpp was built for.
extern const std::uint32_t BLOB_H;
extern const std::uint32_t BLOB_K;
extern const std::uint32_t BLOB_D_IN;
extern const std::uint32_t BLOB_N_LAYERS;
extern const std::size_t   BLOB_LEN;
extern const float         BLOB[];

}
