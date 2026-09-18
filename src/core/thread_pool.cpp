#include "vectordb/core/thread_pool.hpp"
#include <algorithm>

namespace vectordb {

ThreadPool::ThreadPool(size_t threads) {
    if (threads == 0) {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->condition_.wait(lock, [this]() {
                        return this->stop_ || !this->tasks_.empty();
                    });
                    if (this->stop_ && this->tasks_.empty()) {
                        return;
                    }
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                    ++this->active_tasks_;
                }

                task();

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    --this->active_tasks_;
                    if (this->tasks_.empty() && this->active_tasks_ == 0) {
                        this->wait_condition_.notify_all();
                    }
                }
            }
        });
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    wait_condition_.wait(lock, [this]() {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace vectordb
