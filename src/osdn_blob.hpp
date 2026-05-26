#pragma once
//
// On-disk weight blob layout shared by trainer, demo, bit-identity test,
// blob_to_header tool, and ESP32 weight generator.
//
//   bytes 0..3    magic      = 'O','S','D','N'
//   bytes 4..7    version    = uint32 little-endian, currently 1
//   bytes 8..11   H          = uint32 (hidden dim)
//   bytes 12..15  K          = uint32 (OSDN inner rank)
//   bytes 16..19  D_in       = uint32 (input channel count, e.g. 7)
//   bytes 20..23  n_layers   = uint32 (OSDN layers stacked)
//   bytes 24..27  param_count = uint32 (# of float32s that follow)
//   bytes 28..31  reserved   = uint32 (must write 0)
//   bytes 32..    payload    = param_count * float32 little-endian
//
// Param ordering inside the payload mirrors Net::parameters() in
// cgm_train.cpp byte-for-byte; load_blob in osdn_inference.h is the
// canonical decoder.
//
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <istream>
#include <ostream>
#include <string>

namespace osdn_blob {

static constexpr char     MAGIC[4]      = { 'O','S','D','N' };
static constexpr uint32_t CURRENT_VERSION = 1;
static constexpr std::size_t HEADER_BYTES = 32;

struct Header {
    char     magic[4];
    uint32_t version;
    uint32_t H;
    uint32_t K;
    uint32_t D_in;
    uint32_t n_layers;
    uint32_t param_count;
    uint32_t reserved;
};
static_assert(sizeof(Header) == HEADER_BYTES, "Header must be exactly 32 bytes");

inline std::string format_header(const Header& h) {
    std::string out = "magic='";
    out += h.magic[0]; out += h.magic[1]; out += h.magic[2]; out += h.magic[3];
    out += "' version=" + std::to_string(h.version)
         + " H=" + std::to_string(h.H)
         + " K=" + std::to_string(h.K)
         + " D_in=" + std::to_string(h.D_in)
         + " n_layers=" + std::to_string(h.n_layers)
         + " param_count=" + std::to_string(h.param_count);
    return out;
}

inline bool magic_ok(const Header& h) {
    return std::memcmp(h.magic, MAGIC, 4) == 0;
}

// Writes the header to `os`. Does not flush; caller controls that.
inline void write_header(std::ostream& os,
                         uint32_t H, uint32_t K, uint32_t D_in,
                         uint32_t n_layers, uint32_t param_count) {
    Header h;
    std::memcpy(h.magic, MAGIC, 4);
    h.version     = CURRENT_VERSION;
    h.H           = H;
    h.K           = K;
    h.D_in        = D_in;
    h.n_layers    = n_layers;
    h.param_count = param_count;
    h.reserved    = 0;
    os.write(reinterpret_cast<const char*>(&h), sizeof(h));
}

// Reads and validates the header. Returns true iff:
//   - read succeeded (32 bytes available),
//   - magic == "OSDN",
//   - version == CURRENT_VERSION,
//   - all shape fields are positive.
// On true, `out` is populated. On false, `err` carries a human-readable reason.
inline bool read_header(std::istream& is, Header& out, std::string& err) {
    if (!is.read(reinterpret_cast<char*>(&out), sizeof(out))) {
        err = "short read; file too small to contain a header";
        return false;
    }
    if (!magic_ok(out)) {
        err = "magic mismatch — not an OSDN blob (got '"
              + std::string(out.magic, 4) + "')";
        return false;
    }
    if (out.version != CURRENT_VERSION) {
        err = "version mismatch — got " + std::to_string(out.version)
            + ", expected " + std::to_string(CURRENT_VERSION);
        return false;
    }
    if (out.H == 0 || out.K == 0 || out.D_in == 0 || out.n_layers == 0) {
        err = "invalid shape (" + format_header(out) + ")";
        return false;
    }
    if (out.param_count == 0) {
        err = "param_count is zero";
        return false;
    }
    return true;
}

// Comparison helper for readers that have an expected shape from their build.
// Returns true iff every (H, K, D_in, n_layers) field matches. Mismatch details
// are appended to `err` in a printable table.
inline bool shape_matches(const Header& h,
                          uint32_t H, uint32_t K, uint32_t D_in, uint32_t n_layers,
                          std::string& err) {
    bool ok = (h.H == H) && (h.K == K) && (h.D_in == D_in) && (h.n_layers == n_layers);
    if (!ok) {
        err += "  | field      | blob      | expected |\n";
        err += "  | H          | " + std::to_string(h.H)        + " | " + std::to_string(H)        + " |\n";
        err += "  | K          | " + std::to_string(h.K)        + " | " + std::to_string(K)        + " |\n";
        err += "  | D_in       | " + std::to_string(h.D_in)     + " | " + std::to_string(D_in)     + " |\n";
        err += "  | n_layers   | " + std::to_string(h.n_layers) + " | " + std::to_string(n_layers) + " |\n";
    }
    return ok;
}

}  // namespace osdn_blob
