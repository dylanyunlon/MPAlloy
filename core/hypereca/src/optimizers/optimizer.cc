#include "sgd.h"


namespace Dress {

Optimizer* Optimizer::create(const OptimizerOption& opt) {
    Optimizer* o = 0;
    if (opt.type == OptimizerOption::SGD) {
        o = new SGD(opt.dtype, opt.lr);
    }
    if (!o) {
        throw "Unknown optimizer type";
    }
    return o;
}

};  // namespace Dress
