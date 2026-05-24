#pragma once
#include "value.hpp"
#include <complex>

struct ComplexValue {
    ValuePtr re;
    ValuePtr im;

    ComplexValue();
    ComplexValue(ValuePtr r, ValuePtr i);
    ComplexValue(double r, double i);
    explicit ComplexValue(const std::complex<double>& z);

    ComplexValue conj() const;
};

ComplexValue cv(double r, double i);
ComplexValue cv(const ValuePtr& r, const ValuePtr& i);

ComplexValue operator+(const ComplexValue& a, const ComplexValue& b);
ComplexValue operator+(const ComplexValue& a, const ValuePtr& s);
ComplexValue operator+(const ValuePtr& s, const ComplexValue& a);

ComplexValue operator-(const ComplexValue& a, const ComplexValue& b);
ComplexValue operator-(const ComplexValue& a);

ComplexValue operator*(const ComplexValue& a, const ComplexValue& b);
ComplexValue operator*(const ComplexValue& a, const std::complex<double>& z);
ComplexValue operator*(const std::complex<double>& z, const ComplexValue& a);
ComplexValue operator*(const ComplexValue& a, const ValuePtr& s);
ComplexValue operator*(const ValuePtr& s, const ComplexValue& a);
ComplexValue operator*(const ComplexValue& a, double s);
ComplexValue operator*(double s, const ComplexValue& a);

ValuePtr re_of(const ComplexValue& a);
ValuePtr im_of(const ComplexValue& a);
ValuePtr abs2(const ComplexValue& a);
