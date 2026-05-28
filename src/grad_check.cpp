// Finite-difference gradient verification for the Value-based autograd
// primitives in src/value.cpp.
//
// For each op f, autograd-computed dL/dx (with L = f(x), grad seed 1.0)
// is compared against the central finite difference
//   (f(x+h) - f(x-h)) / (2h)
// at several test points. Points labelled "near-singularity" supply an
// explicit analytical expected gradient instead — FD is unreliable when
// |x| approaches the eps clamp window inside vpow.
//
// Test points marked with the "BUG-TRIGGER" comment are constructed to
// exercise findings on the autograd path. They MUST pass after the
// corresponding fix.
//
// Exit status: 0 if every point passes; 1 if any point fails.

#include "value.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool g_all_pass = true;

double pick_h(double x) {
    return std::max(std::fabs(x) * 1e-4, 1e-7);
}

void report(const std::string& label, double x, double g_auto,
            double g_expected, double rel_tol, double abs_tol) {
    double abs_err = std::fabs(g_auto - g_expected);
    double scale   = std::max(std::fabs(g_expected), 1.0);
    double rel_err = abs_err / scale;
    bool pass = (abs_err <= abs_tol) || (rel_err <= rel_tol);
    if (!pass) {
        std::cout << "    FAIL " << label
                  << " @ x=" << std::setprecision(6) << x
                  << "  auto=" << std::setprecision(8) << g_auto
                  << "  expected=" << g_expected
                  << "  abs_err=" << abs_err
                  << "  rel_err=" << rel_err << "\n";
        g_all_pass = false;
    }
}

struct UnaryPoint { double x; double expected; };  // expected NaN -> use FD

void run_unary(const std::string& name,
               std::function<ValuePtr(const ValuePtr&)> op,
               std::function<double(double)> op_scalar,
               const std::vector<UnaryPoint>& pts,
               double rel_tol = 1e-5,
               double abs_tol = 1e-7) {
    int n_run = 0;
    int n_fail_before = 0; (void)n_fail_before;
    bool start_state = g_all_pass;
    for (const auto& p : pts) {
        ValuePtr a = v(p.x);
        ValuePtr out = op(a);
        if (!std::isfinite(out->data)) {
            std::cout << "    skip " << name << " @ x=" << p.x
                      << " (non-finite forward)\n";
            continue;
        }
        out->backward();
        double g_auto = a->grad;
        double g_expected;
        if (std::isnan(p.expected)) {
            double h = pick_h(p.x);
            g_expected = (op_scalar(p.x + h) - op_scalar(p.x - h)) / (2.0 * h);
        } else {
            g_expected = p.expected;
        }
        report(name, p.x, g_auto, g_expected, rel_tol, abs_tol);
        ++n_run;
    }
    if (g_all_pass == start_state) {
        std::cout << "  PASS " << name << " (" << n_run << " pts)\n";
    } else {
        std::cout << "  FAIL " << name << "\n";
    }
}

struct PowPoint { double x; double exponent; double expected; };

void run_vpow(const std::vector<PowPoint>& pts,
              double rel_tol = 1e-5, double abs_tol = 1e-7) {
    int n_run = 0;
    bool start_state = g_all_pass;
    for (const auto& p : pts) {
        ValuePtr a = v(p.x);
        ValuePtr out = vpow(a, p.exponent);
        if (!std::isfinite(out->data)) {
            std::cout << "    skip vpow @ x=" << p.x << " p=" << p.exponent
                      << " (non-finite forward)\n";
            continue;
        }
        out->backward();
        double g_auto = a->grad;
        double g_expected;
        if (std::isnan(p.expected)) {
            double h = pick_h(p.x);
            double fp = std::pow(p.x + h, p.exponent);
            double fm = std::pow(p.x - h, p.exponent);
            g_expected = (fp - fm) / (2.0 * h);
        } else {
            g_expected = p.expected;
        }
        std::string lbl = "vpow(p=" + std::to_string(p.exponent) + ")";
        report(lbl, p.x, g_auto, g_expected, rel_tol, abs_tol);
        ++n_run;
    }
    if (g_all_pass == start_state) {
        std::cout << "  PASS vpow (" << n_run << " pts)\n";
    } else {
        std::cout << "  FAIL vpow\n";
    }
}

struct BinaryPoint { double a; double b; double expected_da; double expected_db; };

void run_binary(const std::string& name,
                std::function<ValuePtr(const ValuePtr&, const ValuePtr&)> op,
                std::function<double(double, double)> op_scalar,
                const std::vector<BinaryPoint>& pts,
                double rel_tol = 1e-5, double abs_tol = 1e-7) {
    int n_run = 0;
    bool start_state = g_all_pass;
    for (const auto& p : pts) {
        // dL/da
        {
            ValuePtr av = v(p.a);
            ValuePtr bv = v(p.b);
            ValuePtr out = op(av, bv);
            if (!std::isfinite(out->data)) {
                std::cout << "    skip " << name << " @ (a=" << p.a
                          << ",b=" << p.b << ") (non-finite forward)\n";
                continue;
            }
            out->backward();
            double g_a = av->grad;
            double exp_da;
            if (std::isnan(p.expected_da)) {
                double h = pick_h(p.a);
                exp_da = (op_scalar(p.a + h, p.b) - op_scalar(p.a - h, p.b))
                         / (2.0 * h);
            } else {
                exp_da = p.expected_da;
            }
            report(name + " dL/da", p.a, g_a, exp_da, rel_tol, abs_tol);
        }
        // dL/db
        {
            ValuePtr av = v(p.a);
            ValuePtr bv = v(p.b);
            ValuePtr out = op(av, bv);
            if (!std::isfinite(out->data)) continue;
            out->backward();
            double g_b = bv->grad;
            double exp_db;
            if (std::isnan(p.expected_db)) {
                double h = pick_h(p.b);
                exp_db = (op_scalar(p.a, p.b + h) - op_scalar(p.a, p.b - h))
                         / (2.0 * h);
            } else {
                exp_db = p.expected_db;
            }
            report(name + " dL/db", p.b, g_b, exp_db, rel_tol, abs_tol);
        }
        ++n_run;
    }
    if (g_all_pass == start_state) {
        std::cout << "  PASS " << name << " (" << n_run << " pts)\n";
    } else {
        std::cout << "  FAIL " << name << "\n";
    }
}

}  // namespace

int main() {
    const double NaN_ = std::numeric_limits<double>::quiet_NaN();
    std::cout << "grad_check — finite-difference vs autograd\n\n";

    run_unary("vexp", vexp,
              [](double x){ return std::exp(x); },
              {{-3.0, NaN_},{-1.0, NaN_},{0.0, NaN_},
               {0.5, NaN_},{1.0, NaN_},{3.0, NaN_}});

    run_unary("vlog", vlog,
              [](double x){ return std::log(x); },
              {{0.1, NaN_},{0.5, NaN_},{1.0, NaN_},
               {2.0, NaN_},{5.0, NaN_},{10.0, NaN_}});

    run_unary("vtanh", vtanh,
              [](double x){ return std::tanh(x); },
              {{-3.0, NaN_},{-1.0, NaN_},{0.0, NaN_},
               {1.0, NaN_},{3.0, NaN_}});

    run_unary("vsigmoid", vsigmoid,
              [](double x){
                  return x >= 0.0
                      ? 1.0 / (1.0 + std::exp(-x))
                      : std::exp(x) / (1.0 + std::exp(x));
              },
              {{-5.0, NaN_},{-1.0, NaN_},{0.0, NaN_},
               {1.0, NaN_},{5.0, NaN_}});

    // vrelu / vabs / vclamp: subgradient ambiguous at kinks; skip those x.
    run_unary("vrelu", vrelu,
              [](double x){ return x > 0.0 ? x : 0.0; },
              {{-2.0, NaN_},{-0.1, NaN_},{0.5, NaN_},{2.0, NaN_}});

    run_unary("vgelu", vgelu,
              [](double x){
                  const double k0 = 0.7978845608028654;
                  const double k1 = 0.044715;
                  return 0.5 * x * (1.0 + std::tanh(k0 * (x + k1 * x * x * x)));
              },
              {{-2.0, NaN_},{-0.5, NaN_},{0.0, NaN_},
               {0.5, NaN_},{2.0, NaN_}});

    run_unary("vabs", vabs,
              [](double x){ return std::fabs(x); },
              {{-2.0, NaN_},{-0.5, NaN_},{0.5, NaN_},{2.0, NaN_}});

    run_unary("vclamp[-1,1]",
              [](const ValuePtr& x){ return vclamp(x, -1.0, 1.0); },
              [](double x){ return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); },
              {{-2.0, NaN_},{-0.5, NaN_},{0.0, NaN_},
               {0.5, NaN_},{2.0, NaN_}});

    // vpow: FD-validatable range first, then BUG-TRIGGER points with
    // analytical expected gradients (FD breaks below ~1e-6).
    run_vpow({
        // FD-validatable, smooth interior of the domain.
        {2.0,  3.0, NaN_},
        {2.0, -1.0, NaN_},
        {2.0,  0.5, NaN_},
        {-2.0, 3.0, NaN_},
        {-2.0, 2.0, NaN_},
        {0.5,  2.0, NaN_},
        {0.5, -2.0, NaN_},
        // BUG-TRIGGER: |x| below eps=1e-12 clamp window. d/dx(x^-1)=-x^-2.
        { 1e-13, -1.0, -1.0e26},
        {-1e-13, -1.0, -1.0e26},
        // Exact zero with well-defined derivative (d/dx x^p at 0 for p>1 is 0).
        { 0.0,    2.0,  0.0},
        { 0.0,    3.0,  0.0},
    });

    run_binary("op+",
               [](const ValuePtr& a, const ValuePtr& b){ return a + b; },
               [](double a, double b){ return a + b; },
               {{1.0, 2.0, NaN_, NaN_},
                {-3.0, 5.0, NaN_, NaN_},
                {0.5, -0.25, NaN_, NaN_}});

    run_binary("op*",
               [](const ValuePtr& a, const ValuePtr& b){ return a * b; },
               [](double a, double b){ return a * b; },
               {{1.0, 2.0, NaN_, NaN_},
                {-3.0, 5.0, NaN_, NaN_},
                {0.5, -0.25, NaN_, NaN_}});

    run_binary("op/",
               [](const ValuePtr& a, const ValuePtr& b){ return a / b; },
               [](double a, double b){ return a / b; },
               {{1.0, 2.0, NaN_, NaN_},
                {-3.0, 5.0, NaN_, NaN_},
                {0.5, -0.25, NaN_, NaN_},
                {2.0, 1.0, NaN_, NaN_}});

    std::cout << "\n"
              << (g_all_pass ? "OVERALL: PASS" : "OVERALL: FAIL")
              << "\n";
    return g_all_pass ? 0 : 1;
}
