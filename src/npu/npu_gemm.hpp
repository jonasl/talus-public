#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "talus_npu.h"

namespace talus::npu {

class NpuGemm {
public:
    struct Config {
        std::string xclbin;
        std::string insts;
        std::uint32_t tokens{4096};
        std::uint32_t k{2560};
        std::uint32_t n{1024};
        std::uint32_t timeout_ms{8000};
        bool isolate{false};
        bool keepalive{true};
        bool allow_unheld_fclk{true};
    };

    static std::unique_ptr<NpuGemm> Create(const Config& config, std::string* error);
    ~NpuGemm();
    NpuGemm(const NpuGemm&) = delete;
    NpuGemm& operator=(const NpuGemm&) = delete;

    std::size_t ABytes() const noexcept;
    std::size_t BBytes() const noexcept;
    std::size_t CBytes() const noexcept;

    int NewBuffer(std::size_t bytes, std::string* error);
    void* Host(int id) const;
    void* Device(int id) const;
    void* Staged(int id) const;

    void FinalizeWeights(int id);
    hipStream_t StageStream() const noexcept;

    bool Submit(hipStream_t ready_stream, int a_id, const int* b_ids, const int* c_ids, int count);
    bool Wait(hipStream_t consumer_stream);

    bool Healthy() const noexcept;
    std::uint32_t tokens() const noexcept;
    std::uint32_t k() const noexcept;
    std::uint32_t n() const noexcept;
    void Stats(talus_npu_stats_t* out) const noexcept;

    static bool CheckEnvSafety(std::string* diag, bool* fclk_held = nullptr);

private:
    NpuGemm();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace talus::npu
