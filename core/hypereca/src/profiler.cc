#include <cstdio>
#include <dress/profiler.h>
#include <dress/config.h>

namespace Dress {
Profiler profiler_instance_;

Profiler* pfri() {
    return &profiler_instance_;
}

bool pfr_enable = cfgi()->getBool("profiler");
int pfr_step = cfgi()->getInt("profile_step");

const std::vector<std::string> all_rows = {"fwd", "bwd"};
const std::vector<std::string> per_thr_rows = {"ard"};

void Profiler::setRank(int r) {
    this->rank = r;
}

void Profiler::addResult(std::string name, int thr, double t) {
    static int rcnt = 0;
    if (pfr_enable) {
        if (name == all_rows[0]) {
            if (rcnt % pfr_step == pfr_step - 1) {
                this->writeResults();
            }
            ++rcnt;
        }
        tr[thr][name].push_back(t);
    }
}

void Profiler::writeResults() {
    if (!pfr_enable) {
        return;
    }
    size_t n_rows = tr[8]["fwd"].size();
    FILE* fout;
    auto filename = cfgi()->getString("profile_out_file");
    if (filename == "") {
        fout = stderr;
#ifdef DRESS_USE_MPI
        if (this->rank > 0) {
            return;
        }
#endif
    } else {
#ifdef DRESS_USE_MPI
        filename += "." + std::to_string(this->rank);
#endif
        fout = fopen(filename.c_str(), "w");
    }
    std::unordered_map<std::string, double> sums;
    for (auto& j: all_rows) {
        fprintf(fout, "%8s", j.c_str());
    }
    for (auto& j: per_thr_rows) {
        for (int k = 0; k < 8; ++k) {
            fprintf(fout, "%6s.%d", j.c_str(), k);
        }
    }
    fprintf(fout, "\n");
    for (size_t i = 0; i < n_rows; ++i) {
        for (auto& j: all_rows) {
            fprintf(fout, "%8.2lf", tr[8][j][i] * 1e3);
            sums[j] += tr[8][j][i];
        }
        for (auto& j: per_thr_rows) {
            for (int k = 0; k < 8; ++k) {
                if (tr[k][j].size() > i) {
                    fprintf(fout, "%8.2lf", tr[k][j][i] * 1e3);
                }
            }
        }
        fprintf(fout, "\n");
    }
    for (auto& j: all_rows) {
        fprintf(fout, "%8.2lf", sums[j] / n_rows * 1e3);
    }
    fprintf(fout, "\n");

    if (filename != "") {
        fclose(fout);
    }
}

};  // namespace Dress
