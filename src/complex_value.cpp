#include "complex_value.hpp"

ComplexValue::ComplexValue() : re(v(0.0)), im(v(0.0)) {}
ComplexValue::ComplexValue(ValuePtr r, ValuePtr i) : re(std::move(r)), im(std::move(i)) {}
ComplexValue::ComplexValue(double r, double i) : re(v(r)), im(v(i)) {}
ComplexValue::ComplexValue(const std::complex<double>& z) : re(v(z.real())), im(v(z.imag())) {}

ComplexValue ComplexValue::conj() const { return ComplexValue(re, -im); }

ComplexValue cv(double r, double i) { return ComplexValue(r, i); }
ComplexValue cv(const ValuePtr& r, const ValuePtr& i) { return ComplexValue(r, i); }

ComplexValue operator+(const ComplexValue& a, const ComplexValue& b) {
    return ComplexValue(a.re + b.re, a.im + b.im);
}
ComplexValue operator+(const ComplexValue& a, const ValuePtr& s) {
    return ComplexValue(a.re + s, a.im);
}
ComplexValue operator+(const ValuePtr& s, const ComplexValue& a) { return a + s; }

ComplexValue operator-(const ComplexValue& a, const ComplexValue& b) {
    return ComplexValue(a.re - b.re, a.im - b.im);
}
ComplexValue operator-(const ComplexValue& a) {
    return ComplexValue(-a.re, -a.im);
}

ComplexValue operator*(const ComplexValue& a, const ComplexValue& b) {
    return ComplexValue(a.re * b.re - a.im * b.im,
                        a.re * b.im + a.im * b.re);
}

ComplexValue operator*(const ComplexValue& a, const std::complex<double>& z) {
    double zr = z.real(), zi = z.imag();
    return ComplexValue(a.re * zr - a.im * zi,
                        a.re * zi + a.im * zr);
}
ComplexValue operator*(const std::complex<double>& z, const ComplexValue& a) { return a * z; }

ComplexValue operator*(const ComplexValue& a, const ValuePtr& s) {
    return ComplexValue(a.re * s, a.im * s);
}
ComplexValue operator*(const ValuePtr& s, const ComplexValue& a) { return a * s; }

ComplexValue operator*(const ComplexValue& a, double s) {
    return ComplexValue(a.re * s, a.im * s);
}
ComplexValue operator*(double s, const ComplexValue& a) { return a * s; }

ValuePtr re_of(const ComplexValue& a) { return a.re; }
ValuePtr im_of(const ComplexValue& a) { return a.im; }
ValuePtr abs2(const ComplexValue& a) { return a.re * a.re + a.im * a.im; }
