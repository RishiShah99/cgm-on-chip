#pragma once
#include "value.hpp"
#include <vector>

std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits);
ValuePtr cross_entropy(const std::vector<ValuePtr>& probs, int label);
