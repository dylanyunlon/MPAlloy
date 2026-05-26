#include <dress/context.h>

#include <set>


namespace Dress {

const int idx_mask = 0xffff;

#ifdef DRESS_USE_CUDA
std::vector<std::vector<cudaStream_t>> streams;
std::set<int> devices;
std::vector<ncclComm_t> comms;

void setCudaStream(int idx, cudaStream_t stream) {
    devices.insert(idx & idx_mask);
    if (streams.size() == 0) {
        int count;
        cudaGetDeviceCount(&count);
        streams.resize(8);
        for (auto& stream: streams) {
            stream.resize(count);
        }
    }
    streams[idx >> 16][idx & idx_mask] = stream;
}

std::vector<int> getCudaDevices() {
    return std::vector(devices.begin(), devices.end());
}

cudaStream_t getCudaStream(int idx) {
    return streams[idx >> 16][idx & idx_mask];
}

void setNcclComm(int idx, ncclComm_t comm) {
    if (comms.size() == 0) {
        int count;
        cudaGetDeviceCount(&count);
        comms.resize(8);
    }
    comms[idx] = comm;
}

ncclComm_t getNcclComm(int idx) {
    return comms[idx];
}
#endif  // DRESS_USE_CUDA
};  // namespace Dress
