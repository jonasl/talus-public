#include "npu_gemm.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace talus::npu {
namespace {

constexpr unsigned kOpcode = 3;
constexpr std::size_t kBytesPer8 = 9;
constexpr std::int64_t kHeartbeatQuietMs = 2000;
constexpr std::size_t kMaxInFlight = 4;
constexpr unsigned kCopyHelpers = 3;
constexpr std::size_t kParallelCopyMin = std::size_t{1} << 20;
constexpr std::size_t kEventPoolSize = 256;
constexpr std::size_t kRunPoolSize = 16;

// XRT BO flag constants matching libxrt_driver_xdna.so.2 (XRT 2.25)
constexpr std::uint32_t kXrtBoCacheable = 1u << 24;
constexpr std::uint32_t kXrtBoShareUserptr = 1u << 25;
constexpr std::uint32_t kXrtBoHostOnly = 1u << 31;
constexpr int kXclBoSyncToDevice = 0;
constexpr int kXclBoSyncFromDevice = 1;
constexpr int kErtCmdStateCompleted = 4;
constexpr int kErtCmdStateTimeout = 5;

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Trace(const char* what, long a = 0, long b = 0) {
    static const int fd = [] {
        const char* path = std::getenv("TALUS_NPU_TRACE");
        return path != nullptr && path[0] != '\0'
                   ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_DSYNC, 0644)
                   : -1;
    }();
    if (fd < 0) return;
    char line[160];
    const int n = std::snprintf(line, sizeof line, "%lld %s %ld %ld\n",
                                static_cast<long long>(NowMs()), what, a, b);
    if (n > 0) (void)!write(fd, line, static_cast<std::size_t>(n));
}

// ABI-compatible pimpl wrappers for libxrt_coreutil.so.2 (zero build-time XRT header dependency)
namespace xrt_abi {
struct Pimpl {
    void* ptr{nullptr};
    void* ctrl{nullptr};
};
struct Uuid {
    std::uint8_t bytes[16]{};
    Uuid() = default;
    Uuid(const Uuid& o) { std::memcpy(bytes, o.bytes, 16); }
    Uuid& operator=(const Uuid& o) {
        std::memcpy(bytes, o.bytes, 16);
        return *this;
    }
    ~Uuid() {}
};

struct Api {
    void* lib{nullptr};
    void (*dev_ctor)(Pimpl*, unsigned int){};
    void (*dev_dtor)(Pimpl*){};
    Uuid (*dev_reg_xclbin)(Pimpl*, const Pimpl*){};
    void (*xcl_ctor_str)(Pimpl*, const std::string&){};
    void (*hwctx_ctor)(Pimpl*, const Pimpl*, const Uuid*, int){};
    void (*hwctx_dtor)(Pimpl*){};
    void (*kern_ctor)(Pimpl*, const Pimpl*, const std::string&){};
    void (*kern_dtor)(Pimpl*){};
    int (*kern_gid)(const Pimpl*, int){};
    void (*bo_ctor_sz)(Pimpl*, const Pimpl*, std::size_t, std::uint32_t, std::uint32_t){};
    void (*bo_ctor_ptr)(Pimpl*, const Pimpl*, void*, std::size_t, std::uint32_t, std::uint32_t){};
    void (*bo_dtor)(Pimpl*){};
    void* (*bo_map)(Pimpl*){};
    void (*bo_sync)(Pimpl*, int, std::size_t, std::size_t){};
    void (*run_ctor)(Pimpl*, const Pimpl*){};
    void (*run_dtor)(Pimpl*){};
    void (*run_set_bo)(Pimpl*, int, const Pimpl*){};
    void (*run_set_scalar)(Pimpl*, int, const void*, std::size_t){};
    void (*run_start)(Pimpl*){};
    int (*run_wait)(const Pimpl*, const std::chrono::milliseconds&){};
    void (*run_abort)(Pimpl*){};

    bool load(std::string* err) {
        lib = dlopen("libxrt_coreutil.so.2", RTLD_NOW | RTLD_GLOBAL);
        if (!lib) {
            lib = dlopen("/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!lib) {
            if (err) *err = std::string("dlopen libxrt_coreutil.so.2 failed: ") + (dlerror() ? dlerror() : "unknown");
            return false;
        }
        auto sym = [&](const char* name, auto& fn) {
            fn = reinterpret_cast<std::decay_t<decltype(fn)>>(dlsym(lib, name));
            if (!fn && err) *err = std::string("missing XRT symbol: ") + name;
            return fn != nullptr;
        };
        return sym("_ZN3xrt6deviceC1Ej", dev_ctor)
            && sym("_ZN3xrt6deviceD1Ev", dev_dtor)
            && sym("_ZN3xrt6device15register_xclbinERKNS_6xclbinE", dev_reg_xclbin)
            && sym("_ZN3xrt6xclbinC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", xcl_ctor_str)
            && sym("_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_4uuidENS0_11access_modeE", hwctx_ctor)
            && sym("_ZN3xrt10hw_contextD1Ev", hwctx_dtor)
            && sym("_ZN3xrt6kernelC1ERKNS_10hw_contextERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", kern_ctor)
            && sym("_ZN3xrt6kernelD1Ev", kern_dtor)
            && sym("_ZNK3xrt6kernel8group_idEi", kern_gid)
            && sym("_ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj", bo_ctor_sz)
            && sym("_ZN3xrt2boC1ERKNS_6deviceEPvmNS0_5flagsEj", bo_ctor_ptr)
            && sym("_ZN3xrt2boD1Ev", bo_dtor)
            && sym("_ZN3xrt2bo3mapEv", bo_map)
            && sym("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm", bo_sync)
            && sym("_ZN3xrt3runC1ERKNS_6kernelE", run_ctor)
            && sym("_ZN3xrt3runD1Ev", run_dtor)
            && sym("_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE", run_set_bo)
            && sym("_ZN3xrt3run16set_arg_at_indexEiPKvm", run_set_scalar)
            && sym("_ZN3xrt3run5startEv", run_start)
            && sym("_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE", run_wait)
            && sym("_ZN3xrt3run5abortEv", run_abort);
    }
};
}  // namespace xrt_abi

struct Call {
    hipEvent_t ready_event{nullptr};
    int a_id{-1};
    std::vector<int> b_ids;
    std::vector<int> c_ids;
    std::uint64_t seq{0};
};

class CopyPool {
public:
    explicit CopyPool(unsigned helpers) : parts_(helpers + 1), next_(parts_) {
        for (unsigned i = 0; i < helpers; ++i) {
            threads_.emplace_back([this] { Run(); });
        }
    }
    ~CopyPool() {
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void Copy(void* dst, const void* src, std::size_t bytes) {
        if (bytes < kParallelCopyMin) {
            std::memcpy(dst, src, bytes);
            return;
        }
        const std::lock_guard serial(call_mutex_);
        {
            const std::lock_guard lock(mutex_);
            dst_ = static_cast<char*>(dst);
            src_ = static_cast<const char*>(src);
            bytes_ = bytes;
            next_ = 0;
            remaining_ = parts_;
        }
        wake_.notify_all();
        Work();
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    void Work() {
        for (;;) {
            unsigned part = 0;
            char* dst = nullptr;
            const char* src = nullptr;
            std::size_t begin = 0, end = 0;
            {
                const std::lock_guard lock(mutex_);
                if (next_ >= parts_) return;
                part = next_++;
                const std::size_t chunk = (bytes_ + parts_ - 1) / parts_;
                begin = std::min<std::size_t>(part * chunk, bytes_);
                end = std::min(begin + chunk, bytes_);
                dst = dst_;
                src = src_;
            }
            std::memcpy(dst + begin, src + begin, end - begin);
            const std::lock_guard lock(mutex_);
            if (--remaining_ == 0) done_.notify_all();
        }
    }
    void Run() {
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || next_ < parts_; });
                if (stop_) return;
            }
            Work();
        }
    }
    const unsigned parts_;
    std::mutex call_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    char* dst_{nullptr};
    const char* src_{nullptr};
    std::size_t bytes_{0};
    unsigned next_;
    unsigned remaining_{0};
    bool stop_{false};
    std::vector<std::thread> threads_;
};

std::string ReadSmallFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
        content.pop_back();
    }
    return content;
}

bool IsFclkHeld(std::string* detail) {
    DIR* dir = opendir("/sys/class/drm");
    if (!dir) {
        if (detail) *detail = "cannot open /sys/class/drm";
        return false;
    }
    std::string dev_dir;
    while (struct dirent* ent = readdir(dir)) {
        if (std::strncmp(ent->d_name, "card", 4) != 0) continue;
        std::string cand = std::string("/sys/class/drm/") + ent->d_name + "/device";
        if (access((cand + "/pp_dpm_fclk").c_str(), R_OK) == 0) {
            dev_dir = cand;
            break;
        }
    }
    closedir(dir);
    if (dev_dir.empty()) {
        if (detail) *detail = "no amdgpu pp_dpm_fclk found";
        return false;
    }
    std::string perf = ReadSmallFile(dev_dir + "/power_dpm_force_performance_level");
    std::ifstream fclk(dev_dir + "/pp_dpm_fclk");
    std::string line, active_line;
    int total_levels = 0;
    while (std::getline(fclk, line)) {
        if (!line.empty()) {
            ++total_levels;
            if (line.find('*') != std::string::npos) active_line = line;
        }
    }
    if (total_levels <= 0) {
        if (detail) *detail = "empty pp_dpm_fclk";
        return false;
    }
    std::string top_prefix = std::to_string(total_levels - 1) + ":";
    bool held = (perf == "manual") && !active_line.empty() && active_line.rfind(top_prefix, 0) == 0;
    if (detail) {
        *detail = "fclk=" + (active_line.empty() ? "unknown" : active_line) + " perf_level=" + perf +
                  (held ? " (held)" : " (NOT held)");
    }
    return held;
}

}  // namespace

struct NpuGemm::Impl {
    Config config;
    xrt_abi::Api xrt;
    xrt_abi::Pimpl device{};
    xrt_abi::Pimpl xclbin{};
    xrt_abi::Pimpl context{};
    xrt_abi::Pimpl kernel{};
    xrt_abi::Pimpl instructions{};
    std::size_t instruction_words{0};
    int group_inst{1};
    int group_a{3};
    int group_b{4};
    int group_c{5};

    struct Buffer {
        void* host{nullptr};
        void* device{nullptr};
        std::size_t bytes{0};
        xrt_abi::Pimpl bo{};
        bool has_bo{false};
        void* private_host{nullptr};
    };
    std::vector<Buffer> buffers;

    hipStream_t stage_stream{nullptr};
    std::vector<hipEvent_t> event_pool;
    std::mutex event_mutex;
    std::size_t event_cursor{0};

    // Pooled reusable xrt::run objects (eliminates per-slice ERT packet allocation overhead)
    std::vector<xrt_abi::Pimpl> run_pool;
    std::deque<std::pair<std::size_t, int>> pending;  // (run_pool_idx, c_buf_id)
    std::vector<xrt_abi::Pimpl> abandoned;

    CopyPool copy_pool{kCopyHelpers};
    std::mutex mutex;

    std::thread worker;
    std::mutex queue_mutex;
    std::condition_variable queue_wake;
    std::condition_variable queue_idle;
    std::deque<Call> queue;
    std::uint64_t enqueued{0};
    std::uint64_t completed{0};
    bool worker_stop{false};
    std::atomic<bool> failed{false};
    std::atomic<std::int64_t> last_submit_ms{0};
    std::chrono::milliseconds run_timeout{8000};
    std::string last_error;

    // Optional hardware-assisted stream wait (hipStreamWaitValue32 on mapped flag)
    using HipStreamWaitValue32Fn = hipError_t (*)(hipStream_t, void*, uint32_t, unsigned int, uint32_t);
    HipStreamWaitValue32Fn hip_stream_wait_u32{nullptr};
    std::uint32_t* done_flag_host{nullptr};
    void* done_flag_dev{nullptr};
    std::atomic<std::uint32_t> submit_seq{0};

    // Telemetry
    bool print_stats{false};
    std::atomic<std::uint64_t> stat_submits{0};
    std::atomic<std::uint64_t> stat_waits{0};
    std::atomic<std::uint64_t> stat_slices{0};
    double copy_a_us{0};
    double launch_us{0};
    double run_wait_us{0};
    double copy_c_us{0};
    std::atomic<std::uint64_t> wait_ns{0};

    // Keepalive heartbeat
    std::thread keepalive;
    std::mutex keepalive_mutex;
    std::condition_variable keepalive_wake;
    bool keepalive_stop{false};
    int heartbeat_a{-1};
    int heartbeat_b{-1};
    int heartbeat_c{-1};

    ~Impl() {
        StopKeepalive();
        StopWorker();
        if (failed.load()) {
            for (auto& r : abandoned) {
                try {
                    if (r.ptr && xrt.run_abort) xrt.run_abort(&r);
                } catch (...) {
                }
            }
            new std::vector<xrt_abi::Pimpl>(std::move(abandoned));
            new std::vector<Buffer>(std::move(buffers));
            return;
        }
        for (auto& r : run_pool) {
            if (r.ptr) xrt.run_dtor(&r);
        }
        run_pool.clear();
        for (auto& b : buffers) {
            if (b.has_bo) {
                xrt.bo_dtor(&b.bo);
                b.has_bo = false;
            }
            if (b.host != nullptr) {
                (void)hipHostFree(b.host);
            }
        }
        buffers.clear();
        if (done_flag_host != nullptr) {
            (void)hipHostFree(done_flag_host);
        }
        for (auto ev : event_pool) {
            if (ev) (void)hipEventDestroy(ev);
        }
        if (stage_stream) (void)hipStreamDestroy(stage_stream);
        if (instructions.ptr) xrt.bo_dtor(&instructions);
        if (kernel.ptr) xrt.kern_dtor(&kernel);
        if (context.ptr) xrt.hwctx_dtor(&context);
        if (xclbin.ptr) reinterpret_cast<std::shared_ptr<void>*>(&xclbin)->~shared_ptr();
        if (device.ptr) xrt.dev_dtor(&device);
    }

    static double Since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
    }

    hipEvent_t NextEvent() {
        const std::lock_guard lock(event_mutex);
        hipEvent_t ev = event_pool[event_cursor % event_pool.size()];
        event_cursor++;
        return ev;
    }

    void CopyToNpu(int id) {
        const auto t0 = std::chrono::steady_clock::now();
        Buffer& b = buffers[id];
        if (config.isolate && b.private_host && b.host) {
            copy_pool.Copy(b.private_host, b.host, b.bytes);
        }
        xrt.bo_sync(&b.bo, kXclBoSyncToDevice, b.bytes, 0);
        copy_a_us += Since(t0);
    }

    void CopyFromNpu(int id) {
        const auto t0 = std::chrono::steady_clock::now();
        Buffer& b = buffers[id];
        xrt.bo_sync(&b.bo, kXclBoSyncFromDevice, b.bytes, 0);
        if (config.isolate && b.private_host && b.host) {
            copy_pool.Copy(b.host, b.private_host, b.bytes);
        }
        copy_c_us += Since(t0);
    }

    bool WaitRun(xrt_abi::Pimpl& run) {
        const int state = xrt.run_wait(&run, run_timeout);
        if (state == kErtCmdStateCompleted) return true;
        failed = true;
        last_error = state == kErtCmdStateTimeout
                         ? "NPU run timed out after " + std::to_string(run_timeout.count()) + " ms"
                         : "NPU run did not complete (state=" + std::to_string(state) + ")";
        std::fprintf(stderr, "talus npu: %s\n", last_error.c_str());
        return false;
    }

    void ReapOldest() {
        auto [ridx, c_id] = pending.front();
        pending.pop_front();
        Trace("reap_begin", c_id);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = WaitRun(run_pool[ridx]);
        run_wait_us += Since(t0);
        if (!ok) {
            abandoned.push_back(run_pool[ridx]);
            run_pool[ridx] = {};
        }
        Trace("reap_done", c_id, ok);
        if (ok) {
            CopyFromNpu(c_id);
        }
    }

    bool Heartbeat() {
        const std::lock_guard lock(mutex);
        if (failed.load() || heartbeat_a < 0) return false;
        if (!pending.empty()) return true;
        if (NowMs() - last_submit_ms.load() < kHeartbeatQuietMs) return true;
        try {
            Trace("heartbeat_fire");
            xrt_abi::Pimpl& r = run_pool[0];
            xrt.run_set_bo(&r, 3, &buffers[heartbeat_a].bo);
            xrt.run_set_bo(&r, 4, &buffers[heartbeat_b].bo);
            xrt.run_set_bo(&r, 5, &buffers[heartbeat_c].bo);
            xrt.run_start(&r);
            return WaitRun(r);
        } catch (const std::exception& e) {
            failed = true;
            last_error = e.what();
            return false;
        }
    }

    void StopKeepalive() {
        {
            const std::lock_guard lock(keepalive_mutex);
            keepalive_stop = true;
        }
        keepalive_wake.notify_all();
        if (keepalive.joinable()) keepalive.join();
    }

    void StopWorker() {
        {
            const std::lock_guard lock(queue_mutex);
            worker_stop = true;
        }
        queue_wake.notify_all();
        if (worker.joinable()) worker.join();
    }

    void WorkerLoop() {
        std::unique_lock lock(queue_mutex);
        for (;;) {
            queue_wake.wait(lock, [this] { return worker_stop || !queue.empty(); });
            if (queue.empty()) return;
            Call call = std::move(queue.front());
            queue.pop_front();
            lock.unlock();
            Process(call);
            lock.lock();
            ++completed;
            if (done_flag_host != nullptr) {
                __atomic_store_n(done_flag_host, static_cast<std::uint32_t>(completed), __ATOMIC_RELEASE);
            }
            queue_idle.notify_all();
        }
    }

    void Process(const Call& call) {
        if (call.ready_event != nullptr) {
            (void)hipEventSynchronize(call.ready_event);
        }
        const std::lock_guard lock(mutex);
        Trace("process_begin", call.a_id, static_cast<long>(call.b_ids.size()));
        last_submit_ms = NowMs();
        if (failed.load()) return;
        try {
            CopyToNpu(call.a_id);
            std::size_t run_cursor = 0;
            for (std::size_t i = 0; i < call.b_ids.size() && !failed.load(); ++i) {
                while (pending.size() >= kMaxInFlight) {
                    ReapOldest();
                }
                const auto t_launch = std::chrono::steady_clock::now();
                const std::size_t ridx = (run_cursor++) % run_pool.size();
                xrt_abi::Pimpl& r = run_pool[ridx];
                xrt.run_set_bo(&r, 3, &buffers[call.a_id].bo);
                xrt.run_set_bo(&r, 4, &buffers[call.b_ids[i]].bo);
                xrt.run_set_bo(&r, 5, &buffers[call.c_ids[i]].bo);
                xrt.run_start(&r);
                pending.emplace_back(ridx, call.c_ids[i]);
                launch_us += Since(t_launch);
                stat_slices.fetch_add(1, std::memory_order_relaxed);
            }
            while (!pending.empty()) {
                ReapOldest();
            }
        } catch (const std::exception& e) {
            failed = true;
            last_error = e.what();
            pending.clear();
        }
        Trace("process_end", call.a_id);
    }

    void WaitTarget(std::uint64_t target) {
        if (target == 0) return;
        Trace("wait_begin", static_cast<long>(target));
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::unique_lock lock(queue_mutex);
            queue_idle.wait(lock, [this, target] { return worker_stop || completed >= target; });
        }
        const auto dt_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        wait_ns.fetch_add(dt_ns, std::memory_order_relaxed);
        const std::uint64_t w = stat_waits.fetch_add(1, std::memory_order_relaxed) + 1;
        Trace("wait_end", static_cast<long>(target));
        if (print_stats && w % 36 == 0) {
            const double wait_us_total = static_cast<double>(wait_ns.load(std::memory_order_relaxed)) / 1000.0;
            std::fprintf(stderr,
                         "talus npu stats: %llu waits, %llu slices | A sync %.0f us, launch %.0f us, NPU wait %.0f us, C sync %.0f us, stream blocked %.0f us (per wait)\n",
                         static_cast<unsigned long long>(w),
                         static_cast<unsigned long long>(stat_slices.load(std::memory_order_relaxed)),
                         copy_a_us / w, launch_us / w, run_wait_us / w, copy_c_us / w, wait_us_total / w);
        }
    }
};

NpuGemm::NpuGemm() = default;
NpuGemm::~NpuGemm() = default;

bool NpuGemm::CheckEnvSafety(std::string* diag, bool* fclk_held) {
    std::string fclk_msg;
    const bool held = IsFclkHeld(&fclk_msg);
    if (fclk_held) *fclk_held = held;
    std::string pmode = ReadSmallFile("/sys/class/accel/accel0/device/power_mode");
    if (pmode.empty()) pmode = "unknown";
    const bool pmode_bad = (pmode == "turbo");
    if (diag) {
        *diag = fclk_msg + ", npu_power_mode=" + pmode;
    }
    return held && !pmode_bad;
}

std::unique_ptr<NpuGemm> NpuGemm::Create(const Config& config, std::string* error) {
    auto fail = [&](const std::string& msg) -> std::unique_ptr<NpuGemm> {
        if (error) *error = msg;
        return nullptr;
    };

    std::string safety_diag;
    bool fclk_held = false;
    (void)CheckEnvSafety(&safety_diag, &fclk_held);
    if (!fclk_held && !config.allow_unheld_fclk) {
        return fail("NPU safety check failed (" + safety_diag +
                    "): require AMD PI 1.0.0.2c+ firmware, or hold GPU fabric clock (sudo tools/npu/fabric-clock.sh hold)");
    }

    std::unique_ptr<NpuGemm> gemm(new NpuGemm());
    gemm->impl_ = std::make_unique<Impl>();
    Impl& i = *gemm->impl_;
    i.config = config;
    i.run_timeout = std::chrono::milliseconds(std::max<std::uint32_t>(1, config.timeout_ms));
    i.print_stats = std::getenv("TALUS_NPU_STATS") != nullptr || std::getenv("GUFO_NPU_STATS") != nullptr;

    std::string xrt_err;
    if (!i.xrt.load(&xrt_err)) {
        return fail(xrt_err);
    }

    try {
        i.xrt.dev_ctor(&i.device, 0u);
        std::string xcl_path = config.xclbin;
        i.xrt.xcl_ctor_str(&i.xclbin, xcl_path);
        xrt_abi::Uuid uuid = i.xrt.dev_reg_xclbin(&i.device, &i.xclbin);
        i.xrt.hwctx_ctor(&i.context, &i.device, &uuid, 0 /* shared */);
        std::string kname = "MLIR_AIE";
        i.xrt.kern_ctor(&i.kernel, &i.context, kname);

        i.group_inst = i.xrt.kern_gid(&i.kernel, 1);
        i.group_a = i.xrt.kern_gid(&i.kernel, 3);
        i.group_b = i.xrt.kern_gid(&i.kernel, 4);
        i.group_c = i.xrt.kern_gid(&i.kernel, 5);

        std::ifstream file(config.insts, std::ios::binary | std::ios::ate);
        if (!file) return fail("NPU instruction file not readable: " + config.insts);
        const auto size = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<char> words(size);
        file.read(words.data(), static_cast<std::streamsize>(size));
        i.instruction_words = size / sizeof(std::uint32_t);
        i.xrt.bo_ctor_sz(&i.instructions, &i.device, size, kXrtBoCacheable, static_cast<std::uint32_t>(i.group_inst));
        std::memcpy(i.xrt.bo_map(&i.instructions), words.data(), size);
        i.xrt.bo_sync(&i.instructions, kXclBoSyncToDevice, size, 0);

        // Pre-allocate pooled xrt::run handles with fixed args 0, 1, 2
        i.run_pool.resize(kRunPoolSize);
        const std::uint64_t inst_words_u64 = i.instruction_words;
        for (auto& r : i.run_pool) {
            i.xrt.run_ctor(&r, &i.kernel);
            i.xrt.run_set_scalar(&r, 0, &kOpcode, sizeof(kOpcode));
            i.xrt.run_set_bo(&r, 1, &i.instructions);
            i.xrt.run_set_scalar(&r, 2, &inst_words_u64, sizeof(inst_words_u64));
        }
    } catch (const std::exception& e) {
        return fail(std::string("NPU setup failed: ") + e.what());
    }

    if (hipStreamCreateWithFlags(&i.stage_stream, hipStreamNonBlocking) != hipSuccess) {
        return fail("hipStreamCreateWithFlags failed for NPU stage stream");
    }
    i.event_pool.resize(kEventPoolSize, nullptr);
    for (auto& ev : i.event_pool) {
        if (hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess) {
            return fail("hipEventCreateWithFlags failed for NPU event pool");
        }
    }

    // Optional hardware wait support via hipStreamWaitValue32 (opt-in when TALUS_NPU_HW_WAIT=1)
    if (const char* hw = std::getenv("TALUS_NPU_HW_WAIT"); hw != nullptr && hw[0] == '1') {
        int dev = 0;
        int can_wait = 0;
        (void)hipGetDevice(&dev);
        (void)hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, dev);
        void* fn = can_wait ? dlsym(RTLD_DEFAULT, "hipStreamWaitValue32") : nullptr;
        if (fn != nullptr) {
            void* flag_h = nullptr;
            void* flag_d = nullptr;
            if (hipHostMalloc(&flag_h, 64, hipHostMallocMapped) == hipSuccess &&
                hipHostGetDevicePointer(&flag_d, flag_h, 0) == hipSuccess) {
                std::memset(flag_h, 0, 64);
                i.done_flag_host = static_cast<std::uint32_t*>(flag_h);
                i.done_flag_dev = flag_d;
                i.hip_stream_wait_u32 = reinterpret_cast<Impl::HipStreamWaitValue32Fn>(fn);
            } else if (flag_h != nullptr) {
                (void)hipHostFree(flag_h);
            }
        }
    }

    // Wake NPU early and validate end-to-end execution
    std::string buf_err;
    i.heartbeat_a = gemm->NewBuffer(gemm->ABytes(), &buf_err);
    i.heartbeat_b = gemm->NewBuffer(gemm->BBytes(), &buf_err);
    i.heartbeat_c = gemm->NewBuffer(gemm->CBytes(), &buf_err);
    if (i.heartbeat_a < 0 || i.heartbeat_b < 0 || i.heartbeat_c < 0) {
        return fail("NPU heartbeat buffers failed: " + buf_err);
    }
    for (int id : {i.heartbeat_a, i.heartbeat_b}) {
        std::memset(i.buffers[id].host, 0, i.buffers[id].bytes);
        i.CopyToNpu(id);
    }
    if (!i.Heartbeat()) {
        return fail("NPU startup heartbeat failed: " + i.last_error);
    }

    Impl* impl = &i;
    i.worker = std::thread([impl] { impl->WorkerLoop(); });
    if (config.keepalive) {
        i.keepalive = std::thread([impl] {
            std::unique_lock lock(impl->keepalive_mutex);
            while (!impl->keepalive_stop) {
                impl->keepalive_wake.wait_for(lock, std::chrono::milliseconds(1000));
                if (impl->keepalive_stop) break;
                lock.unlock();
                impl->Heartbeat();
                lock.lock();
            }
        });
    }
    return gemm;
}

std::size_t NpuGemm::ABytes() const noexcept {
    return std::size_t{impl_->config.tokens} * impl_->config.k / 8 * kBytesPer8;
}
std::size_t NpuGemm::BBytes() const noexcept {
    return std::size_t{impl_->config.k} * impl_->config.n / 8 * kBytesPer8;
}
std::size_t NpuGemm::CBytes() const noexcept {
    return std::size_t{impl_->config.tokens} * impl_->config.n / 8 * kBytesPer8;
}
std::uint32_t NpuGemm::tokens() const noexcept { return impl_->config.tokens; }
std::uint32_t NpuGemm::k() const noexcept { return impl_->config.k; }
std::uint32_t NpuGemm::n() const noexcept { return impl_->config.n; }
bool NpuGemm::Healthy() const noexcept { return !impl_->failed.load(); }
hipStream_t NpuGemm::StageStream() const noexcept { return impl_->stage_stream; }

int NpuGemm::NewBuffer(std::size_t bytes, std::string* error) {
    Impl& i = *impl_;
    void* host = nullptr;
    const unsigned flags = hipHostMallocMapped | (i.config.isolate ? hipHostMallocNonCoherent : 0u);
    if (hipHostMalloc(&host, bytes, flags) != hipSuccess) {
        if (error) *error = "hipHostMalloc failed for NPU buffer";
        return -1;
    }
    Impl::Buffer buf;
    buf.host = host;
    buf.bytes = bytes;
    if (hipHostGetDevicePointer(&buf.device, host, 0) != hipSuccess) {
        (void)hipHostFree(host);
        if (error) *error = "hipHostGetDevicePointer failed for NPU buffer";
        return -1;
    }
    try {
        if (i.config.isolate) {
            i.xrt.bo_ctor_sz(&buf.bo, &i.device, bytes, kXrtBoHostOnly, static_cast<std::uint32_t>(i.group_a));
            buf.has_bo = true;
            buf.private_host = i.xrt.bo_map(&buf.bo);
        } else {
            i.xrt.bo_ctor_ptr(&buf.bo, &i.device, host, bytes, kXrtBoShareUserptr, static_cast<std::uint32_t>(i.group_a));
            buf.has_bo = true;
        }
        i.buffers.push_back(std::move(buf));
    } catch (const std::exception& e) {
        (void)hipHostFree(host);
        if (error) *error = std::string("NPU buffer setup failed: ") + e.what();
        return -1;
    }
    return static_cast<int>(i.buffers.size()) - 1;
}

void* NpuGemm::Host(int id) const { return impl_->buffers.at(id).host; }
void* NpuGemm::Device(int id) const { return impl_->buffers.at(id).device; }
void* NpuGemm::Staged(int id) const { return impl_->buffers.at(id).device; }

void NpuGemm::FinalizeWeights(int id) {
    const std::lock_guard lock(impl_->mutex);
    impl_->CopyToNpu(id);
}

bool NpuGemm::Submit(hipStream_t ready_stream, int a_id, const int* b_ids, const int* c_ids, int count) {
    if (count <= 0) return true;
    Trace("submit", a_id, count);
    hipEvent_t ev = nullptr;
    if (ready_stream != nullptr) {
        ev = impl_->NextEvent();
        if (hipEventRecord(ev, ready_stream) != hipSuccess) return false;
    }
    const std::uint32_t seq = impl_->submit_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    Call call{ev, a_id, {b_ids, b_ids + count}, {c_ids, c_ids + count}, seq};
    {
        const std::lock_guard lock(impl_->queue_mutex);
        impl_->queue.push_back(std::move(call));
        ++impl_->enqueued;
    }
    impl_->queue_wake.notify_one();
    impl_->stat_submits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool NpuGemm::Wait(hipStream_t consumer_stream) {
    Trace("wait_enqueue");
    std::uint64_t target = 0;
    bool already_done = false;
    {
        const std::lock_guard lock(impl_->queue_mutex);
        target = impl_->enqueued;
        already_done = (impl_->completed >= target);
    }
    if (target == 0 || already_done) {
        impl_->stat_waits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (consumer_stream == nullptr) {
        impl_->WaitTarget(target);
        return true;
    }
    if (impl_->hip_stream_wait_u32 != nullptr && impl_->done_flag_dev != nullptr) {
        if (impl_->hip_stream_wait_u32(consumer_stream, impl_->done_flag_dev,
                                       static_cast<std::uint32_t>(target), hipStreamWaitValueGte,
                                       0xFFFFFFFFu) == hipSuccess) {
            impl_->stat_waits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    struct WaitTicket {
        Impl* impl;
        std::uint64_t target;
    };
    auto* ticket = new WaitTicket{impl_.get(), target};
    if (hipLaunchHostFunc(
            consumer_stream,
            [](void* data) {
                std::unique_ptr<WaitTicket> t(static_cast<WaitTicket*>(data));
                t->impl->WaitTarget(t->target);
            },
            ticket) != hipSuccess) {
        delete ticket;
        return false;
    }
    return true;
}

void NpuGemm::Stats(talus_npu_stats_t* out) const noexcept {
    if (!out) return;
    out->submits = impl_->stat_submits.load(std::memory_order_relaxed);
    out->waits = impl_->stat_waits.load(std::memory_order_relaxed);
    out->slices_run = impl_->stat_slices.load(std::memory_order_relaxed);
    out->encode_us_total = impl_->copy_a_us;
    out->npu_wait_us_total = impl_->run_wait_us;
    out->host_wait_us_total = static_cast<double>(impl_->wait_ns.load(std::memory_order_relaxed)) / 1000.0;
    out->copy_us_total = impl_->copy_c_us;
}

}  // namespace talus::npu
