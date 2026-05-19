#include "value.hpp"
#include "loss.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::vector<ValuePtr> logits = { v(2.0), v(1.0), v(0.1) };
    int label = 0;

    auto probs = softmax(logits);
    auto loss  = cross_entropy(probs, label);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "probs = ";
    for (const auto& p : probs) std::cout << p->data << " ";
    std::cout << "\nloss  = " << loss->data << "\n";

    loss->backward();
    std::cout << "grads = ";
    for (const auto& l : logits) std::cout << l->grad << " ";
    std::cout << "\n";

    std::cout << "expected probs = 0.6590 0.2424 0.0986\n";
    std::cout << "expected loss  = 0.4170\n";
    std::cout << "expected grads = -0.3410 0.2424 0.0986\n";
    return 0;
}
