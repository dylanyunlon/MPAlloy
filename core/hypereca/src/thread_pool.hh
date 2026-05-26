#pragma once
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <queue>

namespace Dress {

template<class FType>
class ThreadPool {
private:
    std::vector<std::thread> ths;

    std::mutex mtx;
    std::condition_variable cv;
    bool end;
    std::queue<FType> q;

public:
    ThreadPool(int nth): end(0) {
        for (int i = 0; i < nth; ++i) {
            this->ths.push_back(std::thread(threadFunc, this, i));
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lck(this->mtx);
            this->end = 1;
        }
        this->cv.notify_all();
        for (auto& th: ths) {
            th.join();
        }
    }

    static void threadFunc(ThreadPool* tp, int idx) {
        while (!tp->end) {
            std::unique_lock<std::mutex> lk(tp->mtx);
            tp->cv.wait(lk, [=]{
                return !tp->q.empty() || tp->end;
            });
            if (!tp->q.empty()) {
                auto f = tp->q.front();
                tp->q.pop();
                lk.unlock();
                f();
            } else {
                lk.unlock();
            }
        }
    }

    void push(FType f) {
        {
            std::lock_guard<std::mutex> lck(this->mtx);
            this->q.push(f);
        }
        this->cv.notify_one();
    }
};

};  // namespace Dress
