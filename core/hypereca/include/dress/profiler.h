#pragma once
#ifndef DRESS_PROFILER_H
#define DRESS_PROFILER_H

#include "logging.h"

#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
        std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}

#define TIMESTAMP(__var__) \
    auto __var__ = std::chrono::system_clock::now();

#define TIMER_START(__var__) TIMESTAMP(ts##__var__##begin)
#define TIMER_END(__var__) TIMESTAMP(ts##__var__##end)
#define TIMER_GET(__var__) getDuration(ts##__var__##begin, ts##__var__##end)
#define TIMER_READ(__var__) getDuration(ts##__var__##begin, \
        std::chrono::system_clock::now())

#define REPORT_TIME(__var__) \
    "Time of " << #__var__ << ": " << \
     TIMER_GET(__var__) * 1e3 << " ms; "

#define REPORT_READING(__var__, __name__) \
    "Time " << __name__ << " since " << #__var__ << ": " << \
    TIMER_READ(__var__) * 1e3 << " ms; "

namespace Dress {
class Profiler {
private:
    int rank;
    std::unordered_map<std::string, std::vector<double>> tr[9];
public:
    Profiler() {}
    ~Profiler() {
        this->writeResults();
    }
    void setRank(int r);
    void addResult(std::string name, int thr, double t);
    void writeResults();
};

Profiler* pfri();
};  // namespace Dress

#define PFR_THR_MAIN 8
#define PFR_ADD_TIME(name, thr, t) Dress::pfri()->addResult(name, thr, t)

#endif  // DRESS_PROFILER
