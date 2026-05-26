#include <dress/logging.h>
#include <algorithm>


void printTensorStat(const char* name, float* ptr, size_t size) {
    float sum = 0, max = 0, min = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += std::abs(ptr[i]);
        max = std::max(max, ptr[i]);
        min = std::min(min, ptr[i]);
    }
    PINFO << name << " numel " << size << " mean " << sum / size
        << " max " << max 
        << " min " << min
        << "\n";
}

