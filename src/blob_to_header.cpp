#include <cstdio>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    std::string in_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--in") in_path = next();
        else if (k == "--out") out_path = next();
    }
    if (in_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: blob_to_header --in weights.bin --out osdn_weights.cpp\n");
        return 2;
    }

    std::ifstream f(in_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
    f.seekg(0, std::ios::end);
    std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes <= 0 || bytes % 4 != 0) {
        std::fprintf(stderr, "bad size %lld\n", static_cast<long long>(bytes));
        return 1;
    }
    std::size_t n_floats = static_cast<std::size_t>(bytes) / 4;
    std::vector<float> blob(n_floats);
    f.read(reinterpret_cast<char*>(blob.data()), bytes);
    if (f.gcount() != bytes) { std::fprintf(stderr, "short read\n"); return 1; }

    std::ofstream o(out_path);
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }

    o << "#include \"osdn_weights.h\"\n\n";
    o << "namespace osdn_weights {\n\n";
    o << "const std::size_t BLOB_LEN = " << n_floats << ";\n\n";
    o << "const float BLOB[" << n_floats << "] = {\n";
    o << std::scientific << std::setprecision(9);
    for (std::size_t i = 0; i < n_floats; ++i) {
        if (i % 4 == 0) o << "    ";
        o << blob[i] << "f";
        if (i + 1 < n_floats) o << ",";
        if (i % 4 == 3 || i + 1 == n_floats) o << "\n";
        else o << " ";
    }
    o << "};\n\n";
    o << "}\n";

    std::fprintf(stderr, "wrote %zu floats (%lld bytes) to %s\n",
                 n_floats, static_cast<long long>(bytes), out_path.c_str());
    return 0;
}
