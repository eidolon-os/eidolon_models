#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <dlfcn.h>
#include "cosyvoice2_text_frontend.h"
#include "rkllm.h"
#include "rknn_api.h"

namespace embedded_rkllm {
#define main nexttts_embedded_rkllm_main
#include "cosyvoice2_rkllm_generate_test.cpp"
#undef main
}  // namespace embedded_rkllm

namespace embedded_hift {
#define main nexttts_embedded_hift_main
#include "cosyvoice2_hift_rknn_benchmark.cpp"
#undef main
}  // namespace embedded_hift

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr size_t kChunkTokens = 25;
constexpr size_t kLookaheadTokens = 3;
constexpr size_t kEncoderInputTokens = 28;
constexpr size_t kChunkFrames = 50;
constexpr size_t kPromptTokens = 50;
constexpr size_t kPromptFrames = 100;
constexpr size_t kTokenCache = 256;
constexpr size_t kFlowCacheFrames = 200;
constexpr size_t kRecentFrames = 100;
constexpr size_t kMelChannels = 80;
constexpr size_t kHiddenSize = 896;
constexpr float kCfgRate = 0.7f;

double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Function>
decltype(auto) WithOptionalLock(std::mutex& mutex, bool enabled,
                                double& wait_ms, Function&& function) {
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    if (enabled) {
        const auto wait_start = Clock::now();
        lock.lock();
        wait_ms += Milliseconds(wait_start, Clock::now());
    }
    return std::forward<Function>(function)();
}

void CheckRknn(int result, const std::string& operation) {
    if (result != RKNN_SUCC) {
        throw std::runtime_error(operation + " failed: " +
                                 std::to_string(result));
    }
}

template <typename T>
std::vector<T> ReadBinary(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("failed to open " + path.string());
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary file " + path.string());
    }
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!values.empty()) {
        stream.read(reinterpret_cast<char*>(values.data()), bytes);
    }
    if (!stream) throw std::runtime_error("failed to read " + path.string());
    return values;
}

std::string ReadTextFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("failed to open " + path.string());
    std::string text((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

template <typename T>
void WriteBinary(const fs::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to create " + path.string());
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("failed to write " + path.string());
}

uint16_t FloatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        if ((mantissa & 0x00001000u) != 0) mantissa += 0x00002000u;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if ((mantissa & 0x00001000u) != 0) {
        mantissa += 0x00002000u;
        if ((mantissa & 0x00800000u) != 0) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign |
                                 (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

float HalfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t unbiased = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(unbiased + 127) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

rknn_core_mask ParseCore(const std::string& value) {
    if (value == "auto") return RKNN_NPU_CORE_AUTO;
    if (value == "0") return RKNN_NPU_CORE_0;
    if (value == "1") return RKNN_NPU_CORE_1;
    if (value == "2") return RKNN_NPU_CORE_2;
    if (value == "01") return RKNN_NPU_CORE_0_1;
    if (value == "012") return RKNN_NPU_CORE_0_1_2;
    if (value == "all") return RKNN_NPU_CORE_ALL;
    throw std::runtime_error("invalid NPU core: " + value);
}

size_t ResidentBytes() {
    std::ifstream stream("/proc/self/status");
    std::string key;
    while (stream >> key) {
        if (key == "VmRSS:") {
            size_t kib = 0;
            stream >> kib;
            return kib * 1024;
        }
        std::string rest;
        std::getline(stream, rest);
    }
    return 0;
}

template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::runtime_error("queue capacity is zero");
    }

    bool Push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) return false;
        queue_.push_back(std::move(value));
        max_size_ = std::max(max_size_, queue_.size());
        not_empty_.notify_one();
        return true;
    }

    bool Pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t MaxSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_size_;
    }

   private:
    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
    size_t max_size_ = 0;
};

struct EncoderTiming {
    double prepare_ms = 0.0;
    double run_ms = 0.0;
    double readback_ms = 0.0;
    double cache_rebind_ms = 0.0;
};

class StreamingEncoder {
   public:
    StreamingEncoder(const fs::path& model, rknn_core_mask core) {
        CheckRknn(rknn_init(&context_, const_cast<char*>(model.c_str()), 0, 0,
                            nullptr),
                  "rknn_init(streaming encoder)");
        CheckRknn(rknn_set_core_mask(context_, core),
                  "rknn_set_core_mask(streaming encoder)");
        rknn_input_output_num count{};
        CheckRknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count,
                             sizeof(count)),
                  "RKNN_QUERY_IN_OUT_NUM(streaming encoder)");
        if (count.n_input != 7 || count.n_output != 5) {
            throw std::runtime_error("unexpected streaming encoder IO count");
        }
        inputs_.resize(count.n_input);
        outputs_.resize(count.n_output);
        input_attrs_.resize(count.n_input);
        output_attrs_.resize(count.n_output);
        for (size_t index = 0; index < inputs_.size(); ++index) {
            auto& attr = input_attrs_[index];
            attr.index = index;
            CheckRknn(rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &attr,
                                 sizeof(attr)),
                      "RKNN_QUERY_NATIVE_INPUT_ATTR(streaming encoder)");
            const rknn_tensor_type expected =
                index == 0 ? RKNN_TENSOR_INT32 : RKNN_TENSOR_FLOAT16;
            const size_t bytes = index == 0 ? sizeof(int32_t) : sizeof(uint16_t);
            if (attr.type != expected || attr.fmt != RKNN_TENSOR_UNDEFINED ||
                attr.size_with_stride != attr.n_elems * bytes) {
                throw std::runtime_error("invalid streaming encoder input layout");
            }
            inputs_[index] = rknn_create_mem(context_, attr.size_with_stride);
            if (inputs_[index] == nullptr) throw std::runtime_error("encoder input allocation failed");
            std::memset(inputs_[index]->virt_addr, 0, inputs_[index]->size);
            CheckRknn(rknn_set_io_mem(context_, inputs_[index], &attr),
                      "rknn_set_io_mem(streaming encoder input)");
        }
        for (size_t index = 0; index < outputs_.size(); ++index) {
            auto& attr = output_attrs_[index];
            attr.index = index;
            CheckRknn(rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr,
                                 sizeof(attr)),
                      "RKNN_QUERY_NATIVE_OUTPUT_ATTR(streaming encoder)");
            if (attr.type != RKNN_TENSOR_FLOAT16 ||
                attr.fmt != RKNN_TENSOR_UNDEFINED ||
                attr.size_with_stride != attr.n_elems * sizeof(uint16_t)) {
                throw std::runtime_error("invalid streaming encoder output layout");
            }
            outputs_[index] = rknn_create_mem(context_, attr.size_with_stride);
            if (outputs_[index] == nullptr) throw std::runtime_error("encoder output allocation failed");
            CheckRknn(rknn_set_io_mem(context_, outputs_[index], &attr),
                      "rknn_set_io_mem(streaming encoder output)");
        }
        for (size_t pair = 0; pair < kCacheInputs.size(); ++pair) {
            if (input_attrs_[kCacheInputs[pair]].n_elems !=
                output_attrs_[kCacheOutputs[pair]].n_elems) {
                throw std::runtime_error("streaming encoder cache shape mismatch");
            }
        }
        Reset();
    }

    ~StreamingEncoder() {
        for (auto* memory : inputs_) {
            if (memory != nullptr) rknn_destroy_mem(context_, memory);
        }
        for (auto* memory : outputs_) {
            if (memory != nullptr) rknn_destroy_mem(context_, memory);
        }
        if (context_ != 0) rknn_destroy(context_);
    }

    void Reset() {
        cache_valid_.assign(kTokenCache, 0.0f);
        for (size_t index = 2; index < inputs_.size(); ++index) {
            std::memset(inputs_[index]->virt_addr, 0, inputs_[index]->size);
            CheckRknn(rknn_mem_sync(context_, inputs_[index],
                                    RKNN_MEMORY_SYNC_TO_DEVICE),
                      "reset streaming encoder cache");
        }
    }

    //: cache 全部住在 rknn_create_mem 给的用户侧缓冲里——virt_addr 上的字节就是
    //: 状态本身,rknn_mem_sync 只是 cache 维护,不是驱动侧的隐藏状态。所以存一份
    //: 字节再写回去,和把同样的 token 重喂一遍等价。StreamingFlow 的
    //: ExportState/ImportState 已经在 steady flow 的交接上这么用了。
    struct State {
        std::vector<std::vector<uint8_t>> cache;
        std::vector<float> cache_valid;
    };

    State ExportState() {
        State state;
        state.cache_valid = cache_valid_;
        state.cache.reserve(kCacheInputs.size());
        for (const size_t index : kCacheInputs) {
            //: 这几个缓冲是上一轮 rknn_run 的输出——NPU 写的,所以 CPU 读之前得
            //: 先把缓存行失效掉。
            CheckRknn(rknn_mem_sync(context_, inputs_[index],
                                    RKNN_MEMORY_SYNC_FROM_DEVICE),
                      "export streaming encoder cache");
            const auto* begin =
                static_cast<const uint8_t*>(inputs_[index]->virt_addr);
            state.cache.emplace_back(begin, begin + inputs_[index]->size);
        }
        return state;
    }

    void ImportState(const State& state) {
        if (state.cache.size() != kCacheInputs.size() ||
            state.cache_valid.size() != kTokenCache) {
            throw std::runtime_error("invalid streaming encoder state");
        }
        //: 输入 2 是 cache_valid 的 fp16 副本,每次 Run 都从 cache_valid_ 重填,
        //: 所以恢复标量就够,不用碰那块缓冲。
        cache_valid_ = state.cache_valid;
        for (size_t offset = 0; offset < kCacheInputs.size(); ++offset) {
            const size_t index = kCacheInputs[offset];
            if (state.cache[offset].size() != inputs_[index]->size) {
                throw std::runtime_error("streaming encoder state size mismatch");
            }
            std::memcpy(inputs_[index]->virt_addr, state.cache[offset].data(),
                        state.cache[offset].size());
            CheckRknn(rknn_mem_sync(context_, inputs_[index],
                                    RKNN_MEMORY_SYNC_TO_DEVICE),
                      "import streaming encoder cache");
        }
    }

    std::vector<float> Run(const std::vector<int32_t>& tokens,
                           size_t current, size_t context,
                           EncoderTiming& timing) {
        if (current == 0 || current > kChunkTokens ||
            context > kLookaheadTokens || tokens.size() != current + context) {
            throw std::runtime_error("invalid streaming encoder token chunk");
        }
        const auto prepare_start = Clock::now();
        auto* token_input = static_cast<int32_t*>(inputs_[0]->virt_addr);
        std::fill(token_input, token_input + kEncoderInputTokens, 0);
        std::copy(tokens.begin(), tokens.end(), token_input);
        std::vector<float> token_valid(kEncoderInputTokens, 0.0f);
        std::fill(token_valid.begin(), token_valid.begin() + tokens.size(), 1.0f);
        FillHalf(inputs_[1], token_valid);
        FillHalf(inputs_[2], cache_valid_);
        for (size_t index = 0; index < 3; ++index) {
            CheckRknn(rknn_mem_sync(context_, inputs_[index],
                                    RKNN_MEMORY_SYNC_TO_DEVICE),
                      "sync streaming encoder input");
        }
        const auto run_start = Clock::now();
        timing.prepare_ms = Milliseconds(prepare_start, run_start);
        CheckRknn(rknn_run(context_, nullptr), "rknn_run(streaming encoder)");
        const auto run_end = Clock::now();
        timing.run_ms = Milliseconds(run_start, run_end);
        CheckRknn(rknn_mem_sync(context_, outputs_[0],
                                RKNN_MEMORY_SYNC_FROM_DEVICE),
                  "sync streaming encoder mu");
        std::vector<float> mu(kMelChannels * kChunkFrames);
        const auto* source = static_cast<const uint16_t*>(outputs_[0]->virt_addr);
        for (size_t index = 0; index < mu.size(); ++index) {
            mu[index] = HalfToFloat(source[index]);
        }
        const auto read_end = Clock::now();
        timing.readback_ms = Milliseconds(run_end, read_end);

        const auto rebind_start = Clock::now();
        for (size_t pair = 0; pair < kCacheInputs.size(); ++pair) {
            const size_t input = kCacheInputs[pair];
            const size_t output = kCacheOutputs[pair];
            auto* next_input = outputs_[output];
            auto* next_output = inputs_[input];
            CheckRknn(rknn_set_io_mem(context_, next_input, &input_attrs_[input]),
                      "rebind streaming encoder cache input");
            CheckRknn(rknn_set_io_mem(context_, next_output,
                                      &output_attrs_[output]),
                      "rebind streaming encoder cache output");
            inputs_[input] = next_input;
            outputs_[output] = next_output;
        }
        std::move(cache_valid_.begin() + kChunkTokens, cache_valid_.end(),
                  cache_valid_.begin());
        std::fill(cache_valid_.end() - kChunkTokens, cache_valid_.end(), 0.0f);
        std::fill(cache_valid_.end() - kChunkTokens,
                  cache_valid_.end() - kChunkTokens + current, 1.0f);
        timing.cache_rebind_ms = Milliseconds(rebind_start, Clock::now());
        return mu;
    }

   private:
    void FillHalf(rknn_tensor_mem* memory, const std::vector<float>& values) {
        auto* output = static_cast<uint16_t*>(memory->virt_addr);
        for (size_t index = 0; index < values.size(); ++index) {
            output[index] = FloatToHalf(values[index]);
        }
    }

    //: 每次 Run 之后 cache 输入和输出对调,所以这两组下标一直成对出现。
    static constexpr std::array<size_t, 4> kCacheInputs = {3, 4, 5, 6};
    static constexpr std::array<size_t, 4> kCacheOutputs = {1, 2, 3, 4};

    rknn_context context_ = 0;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_mem*> inputs_;
    std::vector<rknn_tensor_mem*> outputs_;
    std::vector<float> cache_valid_;
};

struct FlowTiming {
    double prepare_ms = 0.0;
    double run_ms = 0.0;
    double readback_ms = 0.0;
    double cache_ms = 0.0;
};

class StreamingFlow {
   public:
    StreamingFlow(const fs::path& model, rknn_core_mask core) {
        CheckRknn(rknn_init(&context_, const_cast<char*>(model.c_str()), 0, 0,
                            nullptr),
                  "rknn_init(streaming Flow)");
        CheckRknn(rknn_set_core_mask(context_, core),
                  "rknn_set_core_mask(streaming Flow)");
        rknn_input_output_num count{};
        CheckRknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count,
                             sizeof(count)),
                  "RKNN_QUERY_IN_OUT_NUM(streaming Flow)");
        if (count.n_input != 14 || count.n_output != 8) {
            throw std::runtime_error("unexpected streaming Flow IO count");
        }
        inputs_.resize(count.n_input);
        outputs_.resize(count.n_output);
        input_attrs_.resize(count.n_input);
        output_attrs_.resize(count.n_output);
        for (size_t index = 0; index < inputs_.size(); ++index) {
            auto& attr = input_attrs_[index];
            attr.index = index;
            CheckRknn(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attr,
                                 sizeof(attr)),
                      "RKNN_QUERY_INPUT_ATTR(streaming Flow)");
            if (attr.type != RKNN_TENSOR_FLOAT16 ||
                attr.fmt != RKNN_TENSOR_UNDEFINED ||
                attr.size_with_stride != attr.n_elems * sizeof(uint16_t)) {
                throw std::runtime_error("invalid streaming Flow input layout");
            }
            inputs_[index] = rknn_create_mem(context_, attr.size_with_stride);
            if (inputs_[index] == nullptr) throw std::runtime_error("Flow input allocation failed");
            std::memset(inputs_[index]->virt_addr, 0, inputs_[index]->size);
            CheckRknn(rknn_set_io_mem(context_, inputs_[index], &attr),
                      "rknn_set_io_mem(streaming Flow input)");
        }
        for (size_t index = 0; index < outputs_.size(); ++index) {
            auto& attr = output_attrs_[index];
            attr.index = index;
            CheckRknn(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr,
                                 sizeof(attr)),
                      "RKNN_QUERY_OUTPUT_ATTR(streaming Flow)");
            if (attr.type != RKNN_TENSOR_FLOAT16 ||
                attr.fmt != RKNN_TENSOR_UNDEFINED ||
                attr.size_with_stride != attr.n_elems * sizeof(uint16_t)) {
                throw std::runtime_error("invalid streaming Flow output layout");
            }
            outputs_[index] = rknn_create_mem(context_, attr.size_with_stride);
            if (outputs_[index] == nullptr) throw std::runtime_error("Flow output allocation failed");
            CheckRknn(rknn_set_io_mem(context_, outputs_[index], &attr),
                      "rknn_set_io_mem(streaming Flow output)");
        }
        constexpr size_t cfg_channels = 2 * kMelChannels;
        if (input_attrs_[0].n_elems % cfg_channels != 0) {
            throw std::runtime_error("invalid streaming Flow sequence shape");
        }
        chunk_frames_ = input_attrs_[0].n_elems / cfg_channels;
        if (chunk_frames_ == 0 ||
            output_attrs_[0].n_elems != 2 * kMelChannels * chunk_frames_) {
            throw std::runtime_error("inconsistent streaming Flow sequence shape");
        }
        Reset();
    }

    ~StreamingFlow() {
        for (auto* memory : inputs_) {
            if (memory != nullptr) rknn_destroy_mem(context_, memory);
        }
        for (auto* memory : outputs_) {
            if (memory != nullptr) rknn_destroy_mem(context_, memory);
        }
        if (context_ != 0) rknn_destroy(context_);
    }

    void Reset() {
        for (size_t index = 6; index <= 12; ++index) {
            std::memset(inputs_[index]->virt_addr, 0, inputs_[index]->size);
            Sync(inputs_[index], RKNN_MEMORY_SYNC_TO_DEVICE,
                 "reset streaming Flow cache");
        }
    }

    size_t ChunkFrames() const { return chunk_frames_; }

    std::vector<std::vector<uint8_t>> ExportState() const {
        std::vector<std::vector<uint8_t>> state;
        state.reserve(7);
        for (size_t index = 6; index <= 12; ++index) {
            const auto* begin =
                static_cast<const uint8_t*>(inputs_[index]->virt_addr);
            state.emplace_back(begin, begin + inputs_[index]->size);
        }
        return state;
    }

    void ImportState(const std::vector<std::vector<uint8_t>>& state) {
        if (state.size() != 7) {
            throw std::runtime_error("invalid streaming Flow state count");
        }
        for (size_t offset = 0; offset < state.size(); ++offset) {
            const size_t index = offset + 6;
            if (state[offset].size() != inputs_[index]->size) {
                throw std::runtime_error("streaming Flow state size mismatch");
            }
            std::memcpy(inputs_[index]->virt_addr, state[offset].data(),
                        state[offset].size());
            Sync(inputs_[index], RKNN_MEMORY_SYNC_TO_DEVICE,
                 "import streaming Flow cache");
        }
    }

    std::vector<float> Run(const std::vector<float>& noise,
                           const std::vector<float>& mu,
                           const std::vector<float>& cond,
                           const std::vector<float>& spks,
                           size_t valid_frames, size_t processed_frames,
                           FlowTiming& timing) {
        const size_t elements = kMelChannels * chunk_frames_;
        if (noise.size() != elements || mu.size() != elements ||
            cond.size() != elements || spks.size() != kMelChannels ||
            valid_frames == 0 || valid_frames > chunk_frames_) {
            throw std::runtime_error("invalid streaming Flow chunk");
        }
        const auto prepare_start = Clock::now();
        std::vector<float> x_cfg(elements * 2);
        std::copy(noise.begin(), noise.end(), x_cfg.begin());
        std::copy(noise.begin(), noise.end(), x_cfg.begin() + elements);
        std::vector<float> mask(chunk_frames_ * 2, 0.0f);
        std::fill(mask.begin(), mask.begin() + valid_frames, 1.0f);
        std::fill(mask.begin() + chunk_frames_,
                  mask.begin() + chunk_frames_ + valid_frames, 1.0f);
        std::vector<float> mu_cfg(elements * 2, 0.0f);
        std::copy(mu.begin(), mu.end(), mu_cfg.begin());
        std::vector<float> times(2, 0.0f);
        std::vector<float> spks_cfg(kMelChannels * 2, 0.0f);
        std::copy(spks.begin(), spks.end(), spks_cfg.begin());
        std::vector<float> cond_cfg(elements * 2, 0.0f);
        std::copy(cond.begin(), cond.end(), cond_cfg.begin());
        const std::array<const std::vector<float>*, 6> values = {
            &x_cfg, &mask, &mu_cfg, &times, &spks_cfg, &cond_cfg};
        for (size_t index = 0; index < values.size(); ++index) {
            FillHalf(inputs_[index], *values[index]);
            Sync(inputs_[index], RKNN_MEMORY_SYNC_TO_DEVICE,
                 "sync streaming Flow input");
        }
        std::vector<float> bias(kFlowCacheFrames, -1.0e4f);
        const size_t valid_cache = std::min(processed_frames, kFlowCacheFrames);
        std::fill(bias.begin(), bias.begin() + valid_cache, 0.0f);
        FillHalf(inputs_[13], bias);
        Sync(inputs_[13], RKNN_MEMORY_SYNC_TO_DEVICE,
             "sync streaming Flow cache bias");
        const auto run_start = Clock::now();
        timing.prepare_ms = Milliseconds(prepare_start, run_start);
        CheckRknn(rknn_run(context_, nullptr), "rknn_run(streaming Flow)");
        const auto run_end = Clock::now();
        timing.run_ms = Milliseconds(run_start, run_end);
        Sync(outputs_[0], RKNN_MEMORY_SYNC_FROM_DEVICE,
             "sync streaming Flow output");
        const auto* estimator =
            static_cast<const uint16_t*>(outputs_[0]->virt_addr);
        std::vector<float> mel(kMelChannels * valid_frames);
        for (size_t channel = 0; channel < kMelChannels; ++channel) {
            for (size_t frame = 0; frame < valid_frames; ++frame) {
                const size_t local = channel * chunk_frames_ + frame;
                const float conditional = HalfToFloat(estimator[local]);
                const float unconditional =
                    HalfToFloat(estimator[elements + local]);
                const float velocity =
                    (1.0f + kCfgRate) * conditional -
                    kCfgRate * unconditional;
                mel[channel * valid_frames + frame] = noise[local] + velocity;
            }
        }
        const auto read_end = Clock::now();
        timing.readback_ms = Milliseconds(run_end, read_end);

        const auto cache_start = Clock::now();
        for (size_t index = 1; index < outputs_.size(); ++index) {
            Sync(outputs_[index], RKNN_MEMORY_SYNC_FROM_DEVICE,
                 "sync streaming Flow cache output");
        }
        CopyCache(6, 1);
        UpdateKv(7, 2, 8, processed_frames, valid_frames);
        CopyCache(8, 3);
        UpdateKv(9, 4, 96, processed_frames, valid_frames);
        CopyCache(10, 5);
        UpdateKv(11, 6, 8, processed_frames, valid_frames);
        CopyCache(12, 7);
        for (size_t index = 6; index <= 12; ++index) {
            Sync(inputs_[index], RKNN_MEMORY_SYNC_TO_DEVICE,
                 "sync streaming Flow cache input");
        }
        timing.cache_ms = Milliseconds(cache_start, Clock::now());
        return mel;
    }

   private:
    void Sync(rknn_tensor_mem* memory, rknn_mem_sync_mode mode,
              const std::string& operation) {
        CheckRknn(rknn_mem_sync(context_, memory, mode), operation);
    }

    void FillHalf(rknn_tensor_mem* memory, const std::vector<float>& values) {
        auto* output = static_cast<uint16_t*>(memory->virt_addr);
        for (size_t index = 0; index < values.size(); ++index) {
            output[index] = FloatToHalf(values[index]);
        }
    }

    void CopyCache(size_t input, size_t output) {
        if (input_attrs_[input].size_with_stride !=
            output_attrs_[output].size_with_stride) {
            throw std::runtime_error("Flow convolution cache size mismatch");
        }
        std::memcpy(inputs_[input]->virt_addr, outputs_[output]->virt_addr,
                    inputs_[input]->size);
    }

    void UpdateKv(size_t input_index, size_t output_index, size_t rows,
                  size_t processed_frames, size_t valid_frames) {
        constexpr size_t width = 1024;
        const size_t input_row = kFlowCacheFrames * width;
        const size_t output_row = chunk_frames_ * width;
        auto* input = static_cast<uint16_t*>(inputs_[input_index]->virt_addr);
        const auto* output =
            static_cast<const uint16_t*>(outputs_[output_index]->virt_addr);
        if (input_attrs_[input_index].n_elems != rows * input_row ||
            output_attrs_[output_index].n_elems != rows * output_row) {
            throw std::runtime_error("Flow KV cache shape mismatch");
        }
        for (size_t row = 0; row < rows; ++row) {
            auto* input_row_ptr = input + row * input_row;
            const auto* output_row_ptr = output + row * output_row;
            if (processed_frames < kPromptFrames) {
                if (processed_frames + valid_frames > kPromptFrames) {
                    throw std::runtime_error(
                        "Flow chunk crosses prompt/target boundary");
                }
                std::memcpy(
                    input_row_ptr + processed_frames * width,
                    output_row_ptr,
                    valid_frames * width * sizeof(uint16_t));
                continue;
            }
            size_t previous_target =
                std::min(processed_frames - kPromptFrames, kRecentFrames);
            const size_t keep_current = std::min(valid_frames, kRecentFrames);
            const size_t current_offset = valid_frames - keep_current;
            if (keep_current == kRecentFrames) {
                std::memcpy(
                    input_row_ptr + kPromptFrames * width,
                    output_row_ptr + current_offset * width,
                    kRecentFrames * width * sizeof(uint16_t));
                continue;
            }
            if (previous_target + keep_current > kRecentFrames) {
                const size_t drop =
                    previous_target + keep_current - kRecentFrames;
                std::memmove(
                    input_row_ptr + kPromptFrames * width,
                    input_row_ptr + (kPromptFrames + drop) * width,
                    (previous_target - drop) * width * sizeof(uint16_t));
                previous_target -= drop;
            }
            std::memcpy(
                input_row_ptr +
                    (kPromptFrames + previous_target) * width,
                output_row_ptr + current_offset * width,
                keep_current * width * sizeof(uint16_t));
        }
    }

    rknn_context context_ = 0;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_mem*> inputs_;
    std::vector<rknn_tensor_mem*> outputs_;
    size_t chunk_frames_ = 0;
};

struct MelChunk {
    size_t index = 0;
    size_t frames = 0;
    bool final = false;
    double ready_ms = 0.0;
    std::vector<float> mel;
};

struct FlowEvent {
    size_t global_chunk = 0;
    size_t current_tokens = 0;
    size_t context_tokens = 0;
    size_t output_frames = 0;
    bool prompt = false;
    bool final = false;
    double start_ms = 0.0;
    double encoder_wait_ms = 0.0;
    double encoder_ms = 0.0;
    double flow_wait_ms = 0.0;
    double flow_ms = 0.0;
    double ready_ms = 0.0;
};

struct PcmEvent {
    size_t call = 0;
    size_t input_frames = 0;
    size_t model_frames = 0;
    size_t playable_samples = 0;
    size_t cumulative_samples = 0;
    double mel_ready_ms = 0.0;
    double start_ms = 0.0;
    double ready_ms = 0.0;
    double f0_wait_ms = 0.0;
    double f0_ms = 0.0;
    double decoder_wait_ms = 0.0;
    double decoder_ms = 0.0;
    double cpu_ms = 0.0;
    double stream_write_ms = 0.0;
    double deadline_margin_ms = 0.0;
    double buffer_after_ms = 0.0;
    bool underrun = false;
};

struct Metrics {
    double cosine = 0.0;
    double mae = 0.0;
    double max_abs = 0.0;
};

Metrics Compare(const std::vector<float>& actual,
                const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("metric tensor size mismatch");
    }
    double dot = 0.0;
    double norm_actual = 0.0;
    double norm_expected = 0.0;
    double error_sum = 0.0;
    double max_error = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double left = actual[index];
        const double right = expected[index];
        const double error = std::abs(left - right);
        dot += left * right;
        norm_actual += left * left;
        norm_expected += right * right;
        error_sum += error;
        max_error = std::max(max_error, error);
    }
    return {dot / std::sqrt(norm_actual * norm_expected),
            error_sum / actual.size(), max_error};
}

std::vector<float> AssembleMel(const std::vector<MelChunk>& chunks) {
    size_t total_frames = 0;
    for (const auto& chunk : chunks) total_frames += chunk.frames;
    std::vector<float> result(kMelChannels * total_frames);
    size_t offset = 0;
    for (const auto& chunk : chunks) {
        for (size_t channel = 0; channel < kMelChannels; ++channel) {
            std::copy(chunk.mel.begin() + channel * chunk.frames,
                      chunk.mel.begin() + (channel + 1) * chunk.frames,
                      result.begin() + channel * total_frames + offset);
        }
        offset += chunk.frames;
    }
    return result;
}

std::vector<float> MakeCond(const std::vector<float>& prompt_feat,
                            size_t global_frame) {
    std::vector<float> cond(kMelChannels * kChunkFrames, 0.0f);
    if (global_frame >= kPromptFrames) return cond;
    for (size_t frame = 0; frame < kChunkFrames; ++frame) {
        for (size_t channel = 0; channel < kMelChannels; ++channel) {
            cond[channel * kChunkFrames + frame] =
                prompt_feat[(global_frame + frame) * kMelChannels + channel];
        }
    }
    return cond;
}

std::vector<float> SliceNoise(const std::vector<float>& chunks,
                              size_t chunk_index) {
    const size_t elements = kMelChannels * kChunkFrames;
    const size_t start = chunk_index * elements;
    if (start + elements > chunks.size()) {
        throw std::runtime_error("noise fixture has too few chunks");
    }
    return std::vector<float>(chunks.begin() + static_cast<ptrdiff_t>(start),
                              chunks.begin() +
                                  static_cast<ptrdiff_t>(start + elements));
}

class FlowNoiseSource {
   public:
    FlowNoiseSource(const std::vector<float>* fixture, uint32_t seed)
        : fixture_(fixture), generator_(seed), distribution_(0.0f, 1.0f) {}

    //: 每句话都从 chunk 0 重新数。生成器本身不复位——同一句说两遍得到不同的
    //: 噪声是对的，而这个守卫要认识的是"新的一句"，不是"乱序"。
    void Reset() { next_chunk_ = 0; }

    std::vector<float> Next(size_t chunk_index) {
        if (chunk_index != next_chunk_) {
            throw std::runtime_error("Flow noise chunks requested out of order");
        }
        ++next_chunk_;
        if (fixture_ != nullptr) return SliceNoise(*fixture_, chunk_index);
        std::vector<float> result(kMelChannels * kChunkFrames);
        for (float& value : result) value = distribution_(generator_);
        return result;
    }

   private:
    const std::vector<float>* fixture_ = nullptr;
    size_t next_chunk_ = 0;
    std::mt19937 generator_;
    std::normal_distribution<float> distribution_;
};

class PcmStreamWriter {
   public:
    //: `-` means the descriptor serve mode set aside for audio, not stdout
    //: itself. librkllm prints its own banner to stdout while it loads, and
    //: 567 bytes of "I rkllm: ..." at the head of a PCM stream is not audio.
    //: The library is not ours, so the service takes the descriptor before the
    //: library can reach it and hands stdout to the journal — see main().
    //: Every other path keeps opening a file exactly as before.
    PcmStreamWriter(const fs::path& path, int audio_fd)
        : path_(path), fd_(path == "-" ? audio_fd : -1) {
        if (path_.empty()) return;
        std::signal(SIGPIPE, SIG_IGN);
        if (path_ == "-") {
            if (fd_ < 0) {
                throw std::runtime_error("no descriptor was reserved for audio");
            }
            return;
        }
        stream_.open(path_, std::ios::binary | std::ios::out);
        if (!stream_) {
            throw std::runtime_error("failed to open PCM stream " +
                                     path_.string());
        }
        sink_ = &stream_;
    }

    bool Enabled() const { return sink_ != nullptr || fd_ >= 0; }
    size_t BytesWritten() const { return bytes_written_; }

    void Write(const std::vector<float>& waveform, size_t begin) {
        if (!Enabled()) return;
        if (begin > waveform.size()) {
            throw std::runtime_error("invalid PCM stream offset");
        }
        std::vector<char> bytes((waveform.size() - begin) * sizeof(int16_t));
        for (size_t index = begin; index < waveform.size(); ++index) {
            const float sample = std::clamp(waveform[index], -1.0f, 1.0f);
            const int16_t pcm = static_cast<int16_t>(std::lrint(
                sample < 0.0f ? sample * 32768.0f : sample * 32767.0f));
            const uint16_t encoded = static_cast<uint16_t>(pcm);
            const size_t output = (index - begin) * sizeof(int16_t);
            bytes[output] = static_cast<char>(encoded & 0xffu);
            bytes[output + 1] = static_cast<char>((encoded >> 8) & 0xffu);
        }
        if (fd_ >= 0) {
            // 一个裸描述符，所以短写要自己接住：管道满的时候 write 只写一部分,
            // 那时候丢掉尾巴就是音频里一个听得见的洞。
            size_t offset = 0;
            while (offset < bytes.size()) {
                const ssize_t written =
                    ::write(fd_, bytes.data() + offset, bytes.size() - offset);
                if (written < 0) {
                    if (errno == EINTR) continue;
                    throw std::runtime_error("failed to write PCM stream");
                }
                offset += static_cast<size_t>(written);
            }
        } else {
            sink_->write(bytes.data(),
                         static_cast<std::streamsize>(bytes.size()));
            sink_->flush();
            if (!*sink_) {
                throw std::runtime_error("failed to write PCM stream " +
                                         path_.string());
            }
        }
        bytes_written_ += bytes.size();
    }

   private:
    fs::path path_;
    std::ofstream stream_;
    std::ostream* sink_ = nullptr;
    int fd_ = -1;
    size_t bytes_written_ = 0;
};

void AppendFrames(std::vector<float>& destination, size_t destination_frames,
                  const std::vector<float>& source, size_t source_frames,
                  size_t valid_frames) {
    if (destination.size() != kMelChannels * destination_frames ||
        source.size() != kMelChannels * source_frames ||
        valid_frames > source_frames) {
        throw std::runtime_error("invalid channel-major frame append");
    }
    std::vector<float> joined(
        kMelChannels * (destination_frames + valid_frames));
    for (size_t channel = 0; channel < kMelChannels; ++channel) {
        std::copy(destination.begin() + channel * destination_frames,
                  destination.begin() + (channel + 1) * destination_frames,
                  joined.begin() +
                      channel * (destination_frames + valid_frames));
        std::copy(source.begin() + channel * source_frames,
                  source.begin() + channel * source_frames + valid_frames,
                  joined.begin() +
                      channel * (destination_frames + valid_frames) +
                      destination_frames);
    }
    destination = std::move(joined);
}

std::vector<float> PadFrames(const std::vector<float>& source,
                             size_t source_frames, size_t output_frames) {
    if (source.size() != kMelChannels * source_frames ||
        source_frames > output_frames) {
        throw std::runtime_error("invalid channel-major frame padding");
    }
    std::vector<float> result(kMelChannels * output_frames, 0.0f);
    for (size_t channel = 0; channel < kMelChannels; ++channel) {
        std::copy(source.begin() + channel * source_frames,
                  source.begin() + (channel + 1) * source_frames,
                  result.begin() + channel * output_frames);
    }
    return result;
}

void WritePipelineJson(const fs::path& path, bool teacher_replay,
                       bool saw_eos, int32_t stop_token,
                       size_t generated_tokens, bool precompute_full_prompt,
                       bool hybrid_steady_flow, bool hift_batch2,
                       bool serialize_downstream_npu,
                       bool dynamic_text_frontend,
                       size_t target_text_tokens, size_t prefill_tokens,
                       double frontend_asset_load_ms,
                       double request_frontend_ms, double profile_ms,
                       double init_ms, double first_token_ms,
                       double llm_done_ms, double first_mel_ms,
                       double flow_done_ms, double first_pcm_ms,
                       double pipeline_done_ms, double audio_seconds,
                       double steady_pcm_rtf, size_t underruns,
                       double min_buffer_ms, size_t rss_peak,
                       size_t token_queue_max, size_t mel_queue_max,
                       bool has_mel_reference, const fs::path& pcm_stream,
                       size_t pcm_stream_bytes,
                       const Metrics& mel_metrics,
                       const std::vector<FlowEvent>& flow_events,
                       const std::vector<PcmEvent>& pcm_events,
                       const fs::path& wav, const fs::path& mel) {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to create " + path.string());
    stream << std::setprecision(10)
           << "{\n  \"status\": \"PASS\",\n"
           << "  \"mode\": \"rkllm_encoder_flow_hift_streaming\",\n"
           << "  \"generation_mode\": \""
           << (teacher_replay ? "teacher_replay" : "sample") << "\",\n"
           << "  \"generated_tokens\": " << generated_tokens << ",\n"
           << "  \"saw_eos\": " << (saw_eos ? "true" : "false") << ",\n"
           << "  \"stop_token\": " << stop_token << ",\n"
           << "  \"precompute_full_prompt\": "
           << (precompute_full_prompt ? "true" : "false") << ",\n"
           << "  \"hybrid_steady_flow\": "
           << (hybrid_steady_flow ? "true" : "false") << ",\n"
           << "  \"hift_batch2\": "
           << (hift_batch2 ? "true" : "false") << ",\n"
           << "  \"serialize_downstream_npu\": "
           << (serialize_downstream_npu ? "true" : "false") << ",\n"
           << "  \"text_frontend_mode\": \""
           << (dynamic_text_frontend ? "edge_dynamic" : "legacy_prefill_file")
           << "\",\n"
           << "  \"target_text_tokens\": " << target_text_tokens << ",\n"
           << "  \"prefill_tokens\": " << prefill_tokens << ",\n"
           << "  \"text_frontend_asset_load_ms\": "
           << frontend_asset_load_ms << ",\n"
           << "  \"request_text_frontend_ms\": " << request_frontend_ms
           << ",\n"
           << "  \"profile_precompute_ms\": " << profile_ms << ",\n"
           << "  \"service_init_warmup_ms\": " << init_ms << ",\n"
           << "  \"first_speech_token_ms\": " << first_token_ms << ",\n"
           << "  \"llm_done_ms\": " << llm_done_ms << ",\n"
           << "  \"first_target_mel_ms\": " << first_mel_ms << ",\n"
           << "  \"flow_done_ms\": " << flow_done_ms << ",\n"
           << "  \"ttft_first_pcm_ms\": " << first_pcm_ms << ",\n"
           << "  \"pipeline_done_ms\": " << pipeline_done_ms << ",\n"
           << "  \"audio_seconds\": " << audio_seconds << ",\n"
           << "  \"request_to_done_rtf\": "
           << pipeline_done_ms / (audio_seconds * 1000.0) << ",\n"
           << "  \"steady_pcm_rtf\": " << steady_pcm_rtf << ",\n"
           << "  \"underrun_count\": " << underruns << ",\n"
           << "  \"minimum_buffer_after_ms\": " << min_buffer_ms << ",\n"
           << "  \"rss_peak_bytes\": " << rss_peak << ",\n"
           << "  \"token_queue_max\": " << token_queue_max << ",\n"
           << "  \"mel_queue_max\": " << mel_queue_max << ",\n"
           << "  \"pcm_stream_bytes\": " << pcm_stream_bytes << ",\n"
           << "  \"pcm_stream_format\": \"s16le/24000/mono\",\n"
           << "  \"pcm_stream_path\": "
           << (pcm_stream.empty() ? "null" : "\"" + pcm_stream.string() + "\"")
           << ",\n  \"mel_vs_host_fixed\": ";
    if (has_mel_reference) {
        stream << "{\"cosine\": " << mel_metrics.cosine
               << ", \"mae\": " << mel_metrics.mae
               << ", \"max_abs\": " << mel_metrics.max_abs << "}";
    } else {
        stream << "null";
    }
    stream << ",\n"
           << "  \"flow_event_count\": " << flow_events.size() << ",\n"
           << "  \"pcm_event_count\": " << pcm_events.size() << ",\n"
           << "  \"output_wav\": \"" << wav.string() << "\",\n"
           << "  \"output_mel\": \"" << mel.string() << "\"\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr
            << "usage: cosyvoice2_streaming_pipeline RKLLM_MODEL HEAD_RKNN "
               "SPEECH_EMBEDDING_F32 ENCODER_RKNN FLOW_RKNN HIFT_MODEL_ROOT "
               "RUNTIME_ROOT FLOW_FIXTURE OUTPUT_ROOT "
               "[--seed=N] [--max-context=N] [--cpu-count=N] [--cpu-mask=N] "
               "[--head-core=MASK] [--encoder-core=MASK] "
               "[--flow-core=MASK] [--hift-core=MASK] "
               "[--threads=N] [--token-queue=N] [--mel-queue=N] "
               "[--sample] [--min-tokens=N] [--max-tokens=N] "
               "[--text=TEXT|--text-file=PATH] "
               "[--text-frontend-root=PATH] [--voice-profile-root=PATH] "
               "[--pcm-stream=PATH] "
               "[--precompute-full-prompt] [--steady-flow-rknn=PATH] "
               "[--hift-batch2] [--serialize-downstream-npu] [--serve]\n";
        return 2;
    }

    LLMHandle llm_handle = nullptr;
    try {
        const fs::path rkllm_model = argv[1];
        const fs::path head_model = argv[2];
        const fs::path speech_embedding_path = argv[3];
        const fs::path encoder_model = argv[4];
        const fs::path flow_model = argv[5];
        const fs::path hift_root = argv[6];
        const fs::path runtime_root = argv[7];
        const fs::path flow_fixture = argv[8];
        const fs::path output_root = argv[9];
        uint32_t seed = 26072000;
        int max_context = 288;
        int cpu_count = 3;
        uint32_t cpu_mask = 0xE0;
        int threads = 4;
        size_t token_queue_capacity = 64;
        size_t mel_queue_capacity = 2;
        int min_tokens = 74;
        size_t max_tokens = 0;
        bool min_tokens_explicit = false;
        bool max_tokens_explicit = false;
        bool teacher_replay = true;
        bool precompute_full_prompt = false;
        bool hift_batch2 = false;
        bool serialize_downstream_npu = false;
        //: Stay resident and synthesize one utterance per line of stdin,
        //: instead of one utterance and exit. Everything expensive — the RKLLM
        //: body, the four RKNN graphs, the HiFT buckets, their warm-ups — is
        //: loaded once and kept: 2.8 s that a per-utterance process pays every
        //: time and a service pays at start. What stays per utterance is the
        //: voice-profile precompute (~0.76 s), because the request consumes the
        //: encoder and Flow cache state it leaves behind.
        bool serve = false;
        fs::path steady_flow_model;
        fs::path pcm_stream_path;
        fs::path text_frontend_root;
        fs::path voice_profile_root;
        std::string target_text;
        bool has_target_text = false;
        std::string head_core_label = "all";
        std::string encoder_core_label = "all";
        std::string flow_core_label = "all";
        std::string hift_core_label = "all";
        for (int index = 10; index < argc; ++index) {
            const std::string option = argv[index];
            if (option.rfind("--seed=", 0) == 0) {
                seed = static_cast<uint32_t>(std::stoul(option.substr(7)));
            } else if (option.rfind("--max-context=", 0) == 0) {
                max_context = std::stoi(option.substr(14));
            } else if (option.rfind("--cpu-count=", 0) == 0) {
                cpu_count = std::stoi(option.substr(12));
            } else if (option.rfind("--cpu-mask=", 0) == 0) {
                cpu_mask = static_cast<uint32_t>(
                    std::stoul(option.substr(11), nullptr, 0));
            } else if (option.rfind("--threads=", 0) == 0) {
                threads = std::stoi(option.substr(10));
            } else if (option.rfind("--token-queue=", 0) == 0) {
                token_queue_capacity = std::stoull(option.substr(14));
            } else if (option.rfind("--mel-queue=", 0) == 0) {
                mel_queue_capacity = std::stoull(option.substr(12));
            } else if (option == "--sample") {
                teacher_replay = false;
            } else if (option == "--teacher-replay") {
                teacher_replay = true;
            } else if (option.rfind("--min-tokens=", 0) == 0) {
                min_tokens = std::stoi(option.substr(13));
                min_tokens_explicit = true;
            } else if (option.rfind("--max-tokens=", 0) == 0) {
                max_tokens = std::stoull(option.substr(13));
                max_tokens_explicit = true;
            } else if (option.rfind("--text=", 0) == 0) {
                if (has_target_text) {
                    throw std::runtime_error("only one text input is allowed");
                }
                target_text = option.substr(7);
                has_target_text = true;
            } else if (option.rfind("--text-file=", 0) == 0) {
                if (has_target_text) {
                    throw std::runtime_error("only one text input is allowed");
                }
                target_text = ReadTextFile(option.substr(12));
                has_target_text = true;
            } else if (option.rfind("--text-frontend-root=", 0) == 0) {
                text_frontend_root = option.substr(21);
            } else if (option.rfind("--voice-profile-root=", 0) == 0) {
                voice_profile_root = option.substr(21);
            } else if (option.rfind("--pcm-stream=", 0) == 0) {
                pcm_stream_path = option.substr(13);
            } else if (option.rfind("--head-core=", 0) == 0) {
                head_core_label = option.substr(12);
            } else if (option.rfind("--encoder-core=", 0) == 0) {
                encoder_core_label = option.substr(15);
            } else if (option.rfind("--flow-core=", 0) == 0) {
                flow_core_label = option.substr(12);
            } else if (option.rfind("--hift-core=", 0) == 0) {
                hift_core_label = option.substr(12);
            } else if (option == "--precompute-full-prompt") {
                precompute_full_prompt = true;
            } else if (option.rfind("--steady-flow-rknn=", 0) == 0) {
                steady_flow_model = option.substr(19);
            } else if (option == "--hift-batch2") {
                hift_batch2 = true;
            } else if (option == "--serialize-downstream-npu") {
                serialize_downstream_npu = true;
            } else if (option == "--serve") {
                serve = true;
            } else {
                throw std::runtime_error("unknown option: " + option);
            }
        }
        if (max_context < 1 || cpu_count < 1 || threads < 1 ||
            min_tokens < 0) {
            throw std::runtime_error("invalid runtime options");
        }
        if (has_target_text && target_text.empty()) {
            throw std::runtime_error("target text is empty");
        }
        if (serve && has_target_text) {
            throw std::runtime_error(
                "--serve takes its text from stdin, not from --text");
        }
        //: "There is text to say" and "text is turned into tokens here" are two
        //: different facts, and serve mode is the case that separates them: it
        //: needs the frontend loaded and has no text yet. Everything that reads
        //: the frontend asks this; only the code that says something now asks
        //: `has_target_text`.
        const bool dynamic_text = has_target_text || serve;
        if (dynamic_text !=
            (!text_frontend_root.empty() && !voice_profile_root.empty())) {
            throw std::runtime_error(
                "dynamic text requires --text-frontend-root and "
                "--voice-profile-root");
        }
        if (!dynamic_text &&
            (!text_frontend_root.empty() || !voice_profile_root.empty())) {
            throw std::runtime_error(
                "text frontend roots require --text or --text-file");
        }
        //: 音频的去处，在任何库被加载之前就定下来。
        //:
        //: librkllm 在加载模型时往 stdout 打自己的横幅，实测 567 字节。对一个
        //: 承诺"stdout 只有 PCM"的服务来说，那是音频流开头的 567 字节噪声。库
        //: 不是我们的，改不了它往哪儿打——所以这里先把 stdout 复制一份留给音频，
        //: 再把 fd 1 指向 stderr：库照它的习惯写 stdout，写进的是 journal。
        int audio_fd = -1;
        if (serve && pcm_stream_path == "-") {
            audio_fd = ::dup(STDOUT_FILENO);
            if (audio_fd < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
                throw std::runtime_error(
                    "could not reserve a descriptor for audio");
            }
        }
        if (serve && teacher_replay) {
            // Teacher replay answers "does this board reproduce the reference
            // mel", which is a fixture question asked once. A service has no
            // reference to replay, and letting the two share a code path would
            // mean the HiFT buckets a request needs depend on that request.
            throw std::runtime_error("--serve cannot replay a teacher fixture");
        }
        fs::create_directories(output_root);

        std::vector<float> prefill;
        std::unique_ptr<cosyvoice2::TextFrontend> text_frontend;
        std::vector<int32_t> planned_target_text_tokens;
        double frontend_asset_load_ms = 0.0;
        if (dynamic_text) {
            const auto frontend_load_start = Clock::now();
            text_frontend = std::make_unique<cosyvoice2::TextFrontend>(
                text_frontend_root, voice_profile_root,
                speech_embedding_path);
            frontend_asset_load_ms =
                Milliseconds(frontend_load_start, Clock::now());
        } else {
            prefill =
                ReadBinary<float>(runtime_root / "rkllm_prefill.f32.bin");
        }
        const fs::path teacher_path = runtime_root / "generated_token.i32.bin";
        const auto teacher = fs::exists(teacher_path)
                                 ? ReadBinary<int32_t>(teacher_path)
                                 : std::vector<int32_t>{};
        const auto prompt = ReadBinary<int32_t>(runtime_root / "flow_prompt_token_50.i32.bin");
        const auto prompt_feat = ReadBinary<float>(runtime_root / "prompt_feat_100.f32.bin");
        const auto speech_embedding = ReadBinary<float>(speech_embedding_path);
        const fs::path noise_path = flow_fixture / "x_chunks.f32.bin";
        const auto noise_chunks =
            teacher_replay && fs::exists(noise_path)
                ? ReadBinary<float>(noise_path)
                : std::vector<float>{};
        const auto spks = ReadBinary<float>(flow_fixture / "spks.f32.bin");
        const fs::path expected_mel_path =
            flow_fixture / "fixed_target_mel.f32.bin";
        const auto expected_mel =
            teacher_replay && fs::exists(expected_mel_path)
                ? ReadBinary<float>(expected_mel_path)
                : std::vector<float>{};
        //: plan_text 与 plan_mel_chunks 的产物。声明在两者之外，因为合成段和
        //: HiFT 桶的选择都要读它们——服务模式下每句被重写一遍。
        size_t planned_prefill_tokens = 0;
        std::vector<size_t> mel_chunk_frames;

        //: 一句话要说多少 token、prefill 占掉多少上下文——只取决于文本，所以
        //: 服务模式下每句重算一遍，而模型一动不动。
        auto plan_text = [&](const std::string& request_text) {
            planned_target_text_tokens =
                dynamic_text ? text_frontend->Tokenize(request_text)
                             : std::vector<int32_t>{};
            planned_prefill_tokens =
                dynamic_text
                    ? text_frontend->PrefillTokenCount(
                          planned_target_text_tokens.size())
                    : prefill.size() / kHiddenSize;
            if ((!dynamic_text &&
                 (prefill.empty() || prefill.size() % kHiddenSize != 0)) ||
                (dynamic_text && planned_prefill_tokens == 0) ||
                prompt.size() != kPromptTokens ||
                prompt_feat.size() != kPromptFrames * kMelChannels ||
                speech_embedding.size() % kHiddenSize != 0 ||
                spks.size() != kMelChannels) {
                throw std::runtime_error("invalid pipeline fixture dimensions");
            }
            if (teacher_replay &&
                (teacher.empty() || noise_chunks.empty() ||
                 expected_mel.size() != teacher.size() * 2 * kMelChannels)) {
                throw std::runtime_error("teacher replay fixtures are incomplete");
            }
            if (dynamic_text && !min_tokens_explicit) {
                const size_t derived_min = planned_target_text_tokens.size() * 2;
                if (derived_min >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("derived minimum token count overflow");
                }
                min_tokens = static_cast<int>(std::max<size_t>(1, derived_min));
            }
            if (dynamic_text && !teacher_replay && !max_tokens_explicit) {
                if (planned_prefill_tokens >= static_cast<size_t>(max_context)) {
                    throw std::runtime_error(
                        "text prefill exhausts the RKLLM context; increase "
                        "--max-context");
                }
                const size_t available =
                    static_cast<size_t>(max_context) - planned_prefill_tokens;
                const size_t target_count = planned_target_text_tokens.size();
                const size_t derived_max =
                    target_count > std::numeric_limits<size_t>::max() / 20
                        ? std::numeric_limits<size_t>::max()
                        : target_count * 20;
                max_tokens = std::min(derived_max, available);
            } else if (max_tokens == 0) {
                if (teacher.empty()) {
                    throw std::runtime_error(
                        "--max-tokens is required without a teacher fixture");
                }
                max_tokens = teacher.size();
            }
            if (dynamic_text &&
                (planned_prefill_tokens >= static_cast<size_t>(max_context) ||
                 max_tokens > static_cast<size_t>(max_context) -
                                  planned_prefill_tokens)) {
                throw std::runtime_error(
                    "prefill plus generation exceeds RKLLM context; increase "
                    "--max-context or reduce --max-tokens");
            }
            if (!teacher_replay && max_tokens <= static_cast<size_t>(min_tokens)) {
                throw std::runtime_error("--max-tokens must exceed --min-tokens");
            }
        };
        if (!serve) plan_text(target_text);

        const size_t speech_vocabulary = speech_embedding.size() / kHiddenSize;
        if (speech_vocabulary < 6561) {
            throw std::runtime_error("speech embedding vocabulary is too small");
        }

        const auto init_start = Clock::now();
        double rkllm_init_ms = 0.0;
        llm_handle = embedded_rkllm::InitializeRkllm(
            rkllm_model.string(), max_context, cpu_count, cpu_mask,
            rkllm_init_ms);
        embedded_rkllm::SpeechHead speech_head(
            head_model.string(), ParseCore(head_core_label), 6561);
        StreamingEncoder encoder(encoder_model,
                                 ParseCore(encoder_core_label));
        StreamingFlow flow(flow_model, ParseCore(flow_core_label));
        std::unique_ptr<StreamingFlow> steady_flow;
        if (!steady_flow_model.empty()) {
            steady_flow = std::make_unique<StreamingFlow>(
                steady_flow_model, ParseCore(flow_core_label));
            if (steady_flow->ChunkFrames() != 2 * kChunkFrames) {
                throw std::runtime_error(
                    "steady Flow model must use 100 Mel frames");
            }
        }

        //: Mel 分块取决于上一步算出的 max_tokens，所以它跟着走。放在模型加载
        //: 之后，因为它要看 steady flow 是否存在。
        auto plan_mel_chunks = [&]() {
            const size_t planned_target_frames =
                (teacher_replay ? teacher.size() : max_tokens) * 2;
            mel_chunk_frames.clear();
            if (hift_batch2 && planned_target_frames > kChunkFrames) {
                mel_chunk_frames.push_back(kChunkFrames);
                size_t remaining = planned_target_frames - kChunkFrames;
                while (remaining > 0) {
                    const size_t fresh = std::min<size_t>(2 * kChunkFrames,
                                                          remaining);
                    mel_chunk_frames.push_back(fresh);
                    remaining -= fresh;
                }
            } else if (steady_flow != nullptr &&
                       planned_target_frames > kChunkFrames) {
                mel_chunk_frames.push_back(kChunkFrames);
                size_t remaining = planned_target_frames - kChunkFrames;
                while (remaining >= 2 * kChunkFrames) {
                    mel_chunk_frames.push_back(2 * kChunkFrames);
                    remaining -= 2 * kChunkFrames;
                }
                while (remaining > 0) {
                    const size_t fresh = std::min(kChunkFrames, remaining);
                    mel_chunk_frames.push_back(fresh);
                    remaining -= fresh;
                }
            } else {
                size_t remaining = planned_target_frames;
                while (remaining > 0) {
                    const size_t fresh =
                        std::min<size_t>(embedded_hift::kNewFrames, remaining);
                    mel_chunk_frames.push_back(fresh);
                    remaining -= fresh;
                }
            }
        };
        if (!serve) plan_mel_chunks();

        std::set<size_t> hift_frames;
        if (teacher_replay) {
            for (size_t call = 0; call < mel_chunk_frames.size(); ++call) {
                hift_frames.insert(
                    mel_chunk_frames[call] +
                    (call == 0 ? 0 : embedded_hift::kContextFrames));
            }
        } else {
            // A sampled EOS can leave any even-sized tail. These sparse buckets
            // cap padding while keeping all RKNN contexts persistent.
            hift_frames = {24, 50, 54, 58};
            if (steady_flow != nullptr || hift_batch2) {
                hift_frames.insert(74);
                hift_frames.insert(108);
            }
        }
        std::map<size_t, std::unique_ptr<embedded_hift::F0Bucket>> f0_buckets;
        std::map<size_t, std::unique_ptr<embedded_hift::DecoderBucket>> decoder_buckets;
        const rknn_core_mask hift_core = ParseCore(hift_core_label);
        for (size_t frames : hift_frames) {
            f0_buckets.emplace(
                frames, std::make_unique<embedded_hift::F0Bucket>(
                            hift_root /
                                ("hift_f0_fp16_seq" + std::to_string(frames) +
                                 ".rknn"),
                            frames, hift_core));
            decoder_buckets.emplace(
                frames, std::make_unique<embedded_hift::DecoderBucket>(
                            hift_root /
                                ("hift_decoder_fp16_mel" +
                                 std::to_string(frames) + ".rknn"),
                            frames, hift_core));
        }
        embedded_hift::StftProcessor stft;
        speech_head.Warmup();
        for (auto& entry : f0_buckets) entry.second->Warmup();
        for (auto& entry : decoder_buckets) entry.second->Warmup();
        FlowNoiseSource flow_noise(teacher_replay ? &noise_chunks : nullptr,
                                   seed);

        // Warm both RKNN graphs, then reset before the voice profile is built.
        EncoderTiming encoder_warm_timing;
        std::vector<int32_t> warm_tokens(kEncoderInputTokens, prompt.front());
        encoder.Run(warm_tokens, kChunkTokens, kLookaheadTokens,
                    encoder_warm_timing);
        encoder.Reset();
        FlowTiming flow_warm_timing;
        flow.Run(std::vector<float>(kMelChannels * kChunkFrames, 0.0f),
                 std::vector<float>(kMelChannels * kChunkFrames, 0.0f),
                 MakeCond(prompt_feat, 0), spks, kChunkFrames, 0,
                 flow_warm_timing);
        flow.Reset();
        if (steady_flow != nullptr) {
            const size_t frames = steady_flow->ChunkFrames();
            FlowTiming steady_warm_timing;
            steady_flow->Run(
                std::vector<float>(kMelChannels * frames, 0.0f),
                std::vector<float>(kMelChannels * frames, 0.0f),
                std::vector<float>(kMelChannels * frames, 0.0f), spks,
                frames, kPromptFrames + kChunkFrames, steady_warm_timing);
            steady_flow->Reset();
        }
        const double init_warmup_ms = Milliseconds(init_start, Clock::now());

        //: 音色预计算一个字的文本都不吃,所以它以前每句重算纯属浪费——重算只是
        //: 因为请求会把 cache 用掉,而当时唯一的复位手段是 Reset(),清零。存一份
        //: 快照再写回去,同样能还原那个起点,不用重跑那 0.8 秒。音色在进程内固定
        //: (--voice-profile-root 是启动参数),所以一份就够整个服务用。
        //:
        //: 有一处不能含糊:预计算并非**只**依赖音色。encoder 那半只吃 prompt
        //: token,是纯音色的;Flow 那半还吃两块噪声,而取噪声的生成器跨句共享、
        //: 故意不复位。所以快照把 prompt 段的噪声钉在了第一句那次抽取上。量过:
        //: 只改这一处时 mel 余弦 0.999975(MAE 约 mel std 的 1%),和基线自己每句
        //: 之间的抖动同一量级——钉住它没有引入新的变化来源。见 README。
        struct VoiceProfile {
            StreamingEncoder::State encoder;
            std::vector<std::vector<uint8_t>> flow;
        };
        std::unique_ptr<VoiceProfile> voice_profile;

        //: 一句话的合成。上面的一切都已加载并预热，这里面是全部随文本变化的
        //: 部分。
        auto synthesize = [&](const std::string& target_text) {
            if (steady_flow != nullptr) steady_flow->Reset();
            flow_noise.Reset();
            // Voice-profile cache: the first prompt chunk and its lookahead are known.
            const auto profile_start = Clock::now();
            if (voice_profile != nullptr) {
                encoder.ImportState(voice_profile->encoder);
                flow.ImportState(voice_profile->flow);
                //: 预计算消费掉的噪声块照样要取走再丢掉。取噪声的生成器是跨句
                //: 共享的——同一句说两遍得到不同的噪声,这是对的——所以少抽这
                //: 两次,后面每一块的噪声都会跟着错位,音频就不再是同一段了。
                flow_noise.Next(0);
                if (precompute_full_prompt) flow_noise.Next(1);
            } else {
                encoder.Reset();
                flow.Reset();
                std::vector<int32_t> profile_tokens(prompt.begin(), prompt.begin() + 28);
                EncoderTiming profile_encoder_timing;
                const auto profile_mu = encoder.Run(
                    profile_tokens, kChunkTokens, kLookaheadTokens,
                    profile_encoder_timing);
                FlowTiming profile_flow_timing;
                flow.Run(flow_noise.Next(0), profile_mu,
                         MakeCond(prompt_feat, 0), spks, kChunkFrames, 0,
                         profile_flow_timing);
                if (precompute_full_prompt) {
                    std::vector<int32_t> remaining_prompt(
                        prompt.begin() + static_cast<ptrdiff_t>(kChunkTokens),
                        prompt.end());
                    EncoderTiming remaining_encoder_timing;
                    const auto remaining_mu = encoder.Run(
                        remaining_prompt, kChunkTokens, 0, remaining_encoder_timing);
                    FlowTiming remaining_flow_timing;
                    flow.Run(flow_noise.Next(1), remaining_mu,
                             MakeCond(prompt_feat, kChunkFrames), spks, kChunkFrames,
                             kChunkFrames, remaining_flow_timing);
                }
                voice_profile = std::make_unique<VoiceProfile>(
                    VoiceProfile{encoder.ExportState(), flow.ExportState()});
            }
            const double profile_ms = Milliseconds(profile_start, Clock::now());
            PcmStreamWriter pcm_stream(pcm_stream_path, audio_fd);

            BoundedQueue<int32_t> token_queue(token_queue_capacity);
            BoundedQueue<MelChunk> mel_queue(mel_queue_capacity);
            std::vector<double> token_ready_ms;
            std::vector<FlowEvent> flow_events;
            std::vector<MelChunk> produced_mel;
            std::vector<PcmEvent> pcm_events;
            std::vector<float> waveform;
            std::vector<int32_t> generated_tokens;
            bool saw_eos = false;
            int32_t stop_token = -1;
            double llm_done_ms = 0.0;
            double flow_done_ms = 0.0;
            double pipeline_done_ms = 0.0;
            std::exception_ptr worker_error;
            std::mutex error_mutex;
            std::mutex downstream_npu_mutex;
            auto record_error = [&](std::exception_ptr value) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (worker_error == nullptr) worker_error = value;
                token_queue.Close();
                mel_queue.Close();
            };

            std::atomic<bool> monitor_running{true};
            std::atomic<size_t> rss_peak{ResidentBytes()};
            std::thread memory_monitor([&] {
                while (monitor_running.load()) {
                    size_t observed = ResidentBytes();
                    size_t previous = rss_peak.load();
                    while (observed > previous &&
                           !rss_peak.compare_exchange_weak(previous, observed)) {
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            });

            const auto request_start = Clock::now();
            double request_frontend_ms = 0.0;
            if (dynamic_text) {
                auto frontend_result = text_frontend->BuildPrefill(target_text);
                request_frontend_ms = Milliseconds(request_start, Clock::now());
                if (frontend_result.target_token_ids !=
                        planned_target_text_tokens ||
                    frontend_result.prefill.size() / kHiddenSize !=
                        planned_prefill_tokens) {
                    throw std::runtime_error(
                        "text frontend changed between planning and request");
                }
                prefill = std::move(frontend_result.prefill);
            }

            std::thread hift_thread([&] {
                try {
                    std::mt19937 generator(seed);
                    std::vector<float> source_cache;
                    std::vector<float> mel_context;
                    embedded_hift::StreamAssembler assembler;
                    size_t call = 0;
                    MelChunk source_chunk;
                    while (mel_queue.Pop(source_chunk)) {
                        if (hift_batch2 && call > 0) {
                            while (source_chunk.frames < 2 * kChunkFrames &&
                                   !source_chunk.final) {
                                MelChunk next_chunk;
                                if (!mel_queue.Pop(next_chunk)) {
                                    throw std::runtime_error(
                                        "HiFT batch ended before the final Mel chunk");
                                }
                                AppendFrames(source_chunk.mel,
                                             source_chunk.frames,
                                             next_chunk.mel, next_chunk.frames,
                                             next_chunk.frames);
                                source_chunk.frames += next_chunk.frames;
                                source_chunk.final = next_chunk.final;
                                source_chunk.ready_ms = next_chunk.ready_ms;
                            }
                        }
                        const auto call_start = Clock::now();
                        std::vector<float> hift_mel;
                        size_t frames = source_chunk.frames;
                        if (call == 0) {
                            hift_mel = source_chunk.mel;
                        } else {
                            frames += embedded_hift::kContextFrames;
                            hift_mel.resize(kMelChannels * frames);
                            if (mel_context.size() !=
                                kMelChannels * embedded_hift::kContextFrames) {
                                throw std::runtime_error("invalid HiFT Mel context");
                            }
                            for (size_t channel = 0; channel < kMelChannels;
                                 ++channel) {
                                std::copy(
                                    mel_context.begin() +
                                        channel * embedded_hift::kContextFrames,
                                    mel_context.begin() +
                                        (channel + 1) *
                                            embedded_hift::kContextFrames,
                                    hift_mel.begin() + channel * frames);
                                std::copy(
                                    source_chunk.mel.begin() +
                                        channel * source_chunk.frames,
                                    source_chunk.mel.begin() +
                                        (channel + 1) * source_chunk.frames,
                                    hift_mel.begin() + channel * frames +
                                        embedded_hift::kContextFrames);
                            }
                        }
                        if (!source_chunk.final &&
                            source_chunk.frames < embedded_hift::kContextFrames) {
                            throw std::runtime_error(
                                "target Mel chunk is too short for overlap cache");
                        }
                        if (source_chunk.frames >= embedded_hift::kContextFrames) {
                            mel_context.resize(kMelChannels *
                                               embedded_hift::kContextFrames);
                            for (size_t channel = 0; channel < kMelChannels;
                                 ++channel) {
                                const auto begin =
                                    source_chunk.mel.begin() +
                                    (channel + 1) * source_chunk.frames -
                                    embedded_hift::kContextFrames;
                                std::copy(
                                    begin,
                                    begin + embedded_hift::kContextFrames,
                                    mel_context.begin() +
                                        channel * embedded_hift::kContextFrames);
                            }
                        }

                        const size_t input_frames = frames;
                        const auto bucket = f0_buckets.lower_bound(input_frames);
                        if (bucket == f0_buckets.end()) {
                            throw std::runtime_error(
                                "no HiFT RKNN bucket can hold final Mel chunk");
                        }
                        const size_t model_frames = bucket->first;
                        if (model_frames != input_frames) {
                            hift_mel = PadFrames(hift_mel, input_frames,
                                                 model_frames);
                        }
                        std::vector<float> pitch;
                        double f0_wait_ms = 0.0;
                        const auto f0_timing = WithOptionalLock(
                            downstream_npu_mutex, serialize_downstream_npu,
                            f0_wait_ms,
                            [&] { return bucket->second->Run(hift_mel, pitch); });
                        const auto cpu_start = Clock::now();
                        auto source = embedded_hift::GenerateNsfSource(
                            pitch, generator, source_cache, threads);
                        std::vector<float> source_stft(
                            embedded_hift::kStftChannels *
                            decoder_buckets.at(model_frames)->StftFrames());
                        stft.Forward(source, source_stft);
                        const auto decoder_start = Clock::now();
                        std::vector<float> decoder_raw;
                        double decoder_wait_ms = 0.0;
                        const auto decoder_timing = WithOptionalLock(
                            downstream_npu_mutex, serialize_downstream_npu,
                            decoder_wait_ms, [&] {
                                return decoder_buckets.at(model_frames)->Run(
                                    hift_mel, source_stft, decoder_raw);
                            });
                        const auto decoder_end = Clock::now();
                        auto call_waveform = stft.Inverse(
                            decoder_raw,
                            decoder_buckets.at(model_frames)->StftFrames());
                        const size_t input_samples =
                            input_frames * embedded_hift::kF0Upsample;
                        if (call_waveform.size() < input_samples) {
                            throw std::runtime_error(
                                "HiFT output is shorter than the valid Mel input");
                        }
                        call_waveform.resize(input_samples);
                        const size_t playable_before = waveform.size();
                        assembler.Append(std::move(call_waveform),
                                         source_chunk.final, waveform);
                        source_cache = embedded_hift::Tail(
                            source, embedded_hift::kSourceCacheSamples);
                        const auto stream_write_start = Clock::now();
                        pcm_stream.Write(waveform, playable_before);
                        const auto ready = Clock::now();
                        PcmEvent event;
                        event.call = call;
                        event.input_frames = input_frames;
                        event.model_frames = model_frames;
                        event.playable_samples = waveform.size() - playable_before;
                        event.cumulative_samples = waveform.size();
                        event.mel_ready_ms = source_chunk.ready_ms;
                        event.start_ms = Milliseconds(request_start, call_start);
                        event.ready_ms = Milliseconds(request_start, ready);
                        event.f0_wait_ms = f0_wait_ms;
                        event.f0_ms = f0_timing.npu_ms;
                        event.decoder_wait_ms = decoder_wait_ms;
                        event.decoder_ms = decoder_timing.npu_ms;
                        event.cpu_ms =
                            Milliseconds(cpu_start, decoder_start) +
                            Milliseconds(decoder_end, stream_write_start);
                        event.stream_write_ms =
                            Milliseconds(stream_write_start, ready);
                        if (!pcm_events.empty()) {
                            const double deadline = pcm_events.front().ready_ms +
                                1000.0 * playable_before /
                                    embedded_hift::kSampleRate;
                            event.deadline_margin_ms = deadline - event.ready_ms;
                            event.underrun = event.deadline_margin_ms < 0.0;
                        }
                        event.buffer_after_ms =
                            pcm_events.empty()
                                ? 1000.0 * event.cumulative_samples /
                                      embedded_hift::kSampleRate
                                : event.deadline_margin_ms +
                                      1000.0 * event.playable_samples /
                                          embedded_hift::kSampleRate;
                        pcm_events.push_back(event);
                        ++call;
                    }
                    pipeline_done_ms = Milliseconds(request_start, Clock::now());
                } catch (...) {
                    record_error(std::current_exception());
                }
            });

            std::thread flow_thread([&] {
                try {
                    std::vector<int32_t> available = prompt;
                    size_t next_start = precompute_full_prompt
                                            ? kPromptTokens
                                            : kChunkTokens;
                    size_t global_chunk = precompute_full_prompt ? 2 : 1;
                    bool producer_closed = false;
                    bool steady_state_ready = false;
                    struct PendingPart {
                        size_t global_chunk = 0;
                        size_t current_tokens = 0;
                        size_t context_tokens = 0;
                        size_t frames = 0;
                        size_t processed_frames = 0;
                        double start_ms = 0.0;
                        double encoder_wait_ms = 0.0;
                        double encoder_ms = 0.0;
                        std::vector<float> mu;
                        std::vector<float> noise;
                    };
                    std::vector<PendingPart> pending_parts;
                    std::vector<float> pending_mu;
                    std::vector<float> pending_noise;
                    size_t pending_frames = 0;
                    while (true) {
                        const size_t remaining = available.size() - next_start;
                        bool can_run = remaining >= kChunkTokens + kLookaheadTokens;
                        if (producer_closed && remaining > 0) can_run = true;
                        if (!can_run) {
                            int32_t token = 0;
                            if (token_queue.Pop(token)) {
                                available.push_back(token);
                                continue;
                            }
                            producer_closed = true;
                            if (available.size() == next_start) break;
                            continue;
                        }
                        const auto event_start = Clock::now();
                        const size_t now_remaining = available.size() - next_start;
                        const size_t current =
                            std::min(kChunkTokens, now_remaining);
                        const size_t context =
                            std::min(kLookaheadTokens, now_remaining - current);
                        const bool final =
                            producer_closed && next_start + current == available.size();
                        std::vector<int32_t> encoder_tokens(
                            available.begin() + static_cast<ptrdiff_t>(next_start),
                            available.begin() + static_cast<ptrdiff_t>(
                                next_start + current + context));
                        EncoderTiming encoder_timing;
                        double encoder_wait_ms = 0.0;
                        auto mu = WithOptionalLock(
                            downstream_npu_mutex, serialize_downstream_npu,
                            encoder_wait_ms, [&] {
                                return encoder.Run(encoder_tokens, current, context,
                                                   encoder_timing);
                            });
                        const size_t valid_frames = current * 2;
                        const double encoder_ms =
                            encoder_timing.prepare_ms + encoder_timing.run_ms +
                            encoder_timing.readback_ms +
                            encoder_timing.cache_rebind_ms;
                        const bool prompt_chunk = next_start < kPromptTokens;
                        const bool use_steady =
                            steady_flow != nullptr && !prompt_chunk &&
                            !produced_mel.empty();
                        if (use_steady) {
                            const auto noise = flow_noise.Next(global_chunk);
                            PendingPart part;
                            part.global_chunk = global_chunk;
                            part.current_tokens = current;
                            part.context_tokens = context;
                            part.frames = valid_frames;
                            part.processed_frames = next_start * 2;
                            part.start_ms =
                                Milliseconds(request_start, event_start);
                            part.encoder_wait_ms = encoder_wait_ms;
                            part.encoder_ms = encoder_ms;
                            part.mu = mu;
                            part.noise = noise;
                            pending_parts.push_back(std::move(part));
                            AppendFrames(pending_mu, pending_frames, mu,
                                         kChunkFrames, valid_frames);
                            AppendFrames(pending_noise, pending_frames, noise,
                                         kChunkFrames, valid_frames);
                            pending_frames += valid_frames;
                            next_start += current;
                            ++global_chunk;
                            if (pending_frames < steady_flow->ChunkFrames() &&
                                !final) {
                                continue;
                            }
                            if (!steady_state_ready) {
                                steady_flow->ImportState(flow.ExportState());
                                steady_state_ready = true;
                            }
                            if (pending_frames == steady_flow->ChunkFrames()) {
                                FlowTiming flow_timing;
                                double flow_wait_ms = 0.0;
                                auto mel = WithOptionalLock(
                                    downstream_npu_mutex,
                                    serialize_downstream_npu, flow_wait_ms, [&] {
                                        return steady_flow->Run(
                                            pending_noise, pending_mu,
                                            std::vector<float>(
                                                kMelChannels *
                                                steady_flow->ChunkFrames(),
                                                0.0f),
                                            spks, pending_frames,
                                            pending_parts.front().processed_frames,
                                            flow_timing);
                                    });
                                const double ready_ms =
                                    Milliseconds(request_start, Clock::now());
                                FlowEvent event;
                                event.global_chunk =
                                    pending_parts.front().global_chunk;
                                for (const auto& pending : pending_parts) {
                                    event.current_tokens += pending.current_tokens;
                                    event.encoder_wait_ms +=
                                        pending.encoder_wait_ms;
                                    event.encoder_ms += pending.encoder_ms;
                                }
                                event.context_tokens =
                                    pending_parts.back().context_tokens;
                                event.output_frames = pending_frames;
                                event.prompt = false;
                                event.final = final;
                                event.start_ms = pending_parts.front().start_ms;
                                event.flow_wait_ms = flow_wait_ms;
                                event.flow_ms =
                                    flow_timing.prepare_ms + flow_timing.run_ms +
                                    flow_timing.readback_ms + flow_timing.cache_ms;
                                event.ready_ms = ready_ms;
                                flow_events.push_back(event);
                                MelChunk chunk;
                                chunk.index = produced_mel.size();
                                chunk.frames = pending_frames;
                                chunk.final = final;
                                chunk.ready_ms = ready_ms;
                                chunk.mel = std::move(mel);
                                produced_mel.push_back(chunk);
                                if (!mel_queue.Push(std::move(chunk))) break;
                            } else {
                                // Never pad a partial tail into the 100-frame graph:
                                // its last attention block changes valid-frame math.
                                flow.ImportState(steady_flow->ExportState());
                                bool queue_open = true;
                                for (size_t index = 0;
                                     index < pending_parts.size(); ++index) {
                                    auto& pending = pending_parts[index];
                                    FlowTiming flow_timing;
                                    double flow_wait_ms = 0.0;
                                    auto mel = WithOptionalLock(
                                        downstream_npu_mutex,
                                        serialize_downstream_npu, flow_wait_ms, [&] {
                                            return flow.Run(
                                                pending.noise, pending.mu,
                                                std::vector<float>(
                                                    kMelChannels * kChunkFrames,
                                                    0.0f),
                                                spks, pending.frames,
                                                pending.processed_frames,
                                                flow_timing);
                                        });
                                    const double ready_ms = Milliseconds(
                                        request_start, Clock::now());
                                    const bool part_final =
                                        final && index + 1 == pending_parts.size();
                                    FlowEvent event;
                                    event.global_chunk = pending.global_chunk;
                                    event.current_tokens = pending.current_tokens;
                                    event.context_tokens = pending.context_tokens;
                                    event.output_frames = pending.frames;
                                    event.prompt = false;
                                    event.final = part_final;
                                    event.start_ms = pending.start_ms;
                                    event.encoder_wait_ms =
                                        pending.encoder_wait_ms;
                                    event.encoder_ms = pending.encoder_ms;
                                    event.flow_wait_ms = flow_wait_ms;
                                    event.flow_ms =
                                        flow_timing.prepare_ms +
                                        flow_timing.run_ms +
                                        flow_timing.readback_ms +
                                        flow_timing.cache_ms;
                                    event.ready_ms = ready_ms;
                                    flow_events.push_back(event);
                                    MelChunk chunk;
                                    chunk.index = produced_mel.size();
                                    chunk.frames = pending.frames;
                                    chunk.final = part_final;
                                    chunk.ready_ms = ready_ms;
                                    chunk.mel = std::move(mel);
                                    produced_mel.push_back(chunk);
                                    if (!mel_queue.Push(std::move(chunk))) {
                                        queue_open = false;
                                        break;
                                    }
                                }
                                if (!queue_open) break;
                            }
                            pending_parts.clear();
                            pending_mu.clear();
                            pending_noise.clear();
                            pending_frames = 0;
                            continue;
                        }

                        FlowTiming flow_timing;
                        double flow_wait_ms = 0.0;
                        auto mel = WithOptionalLock(
                            downstream_npu_mutex, serialize_downstream_npu,
                            flow_wait_ms, [&] {
                                return flow.Run(
                                    flow_noise.Next(global_chunk), mu,
                                    MakeCond(prompt_feat, next_start * 2), spks,
                                    valid_frames, next_start * 2, flow_timing);
                            });
                        const double ready_ms =
                            Milliseconds(request_start, Clock::now());
                        FlowEvent event;
                        event.global_chunk = global_chunk;
                        event.current_tokens = current;
                        event.context_tokens = context;
                        event.output_frames = valid_frames;
                        event.prompt = prompt_chunk;
                        event.final = final;
                        event.start_ms = Milliseconds(request_start, event_start);
                        event.encoder_wait_ms = encoder_wait_ms;
                        event.encoder_ms = encoder_ms;
                        event.flow_wait_ms = flow_wait_ms;
                        event.flow_ms =
                            flow_timing.prepare_ms + flow_timing.run_ms +
                            flow_timing.readback_ms + flow_timing.cache_ms;
                        event.ready_ms = ready_ms;
                        flow_events.push_back(event);
                        if (!event.prompt) {
                            MelChunk chunk;
                            chunk.index = produced_mel.size();
                            chunk.frames = valid_frames;
                            chunk.final = final;
                            chunk.ready_ms = ready_ms;
                            chunk.mel = std::move(mel);
                            produced_mel.push_back(chunk);
                            if (!mel_queue.Push(std::move(chunk))) break;
                            if (steady_flow != nullptr && !steady_state_ready) {
                                steady_flow->ImportState(flow.ExportState());
                                steady_state_ready = true;
                            }
                        }
                        next_start += current;
                        ++global_chunk;
                    }
                    flow_done_ms = Milliseconds(request_start, Clock::now());
                    mel_queue.Close();
                } catch (...) {
                    record_error(std::current_exception());
                }
            });

            std::thread llm_thread([&] {
                try {
                    auto hidden = embedded_rkllm::RunHidden(
                        llm_handle, prefill.data(), prefill.size() / kHiddenSize,
                        true);
                    std::mt19937 generator(seed);
                    std::vector<int32_t> decoded;
                    std::vector<float> logits;
                    const size_t generation_limit =
                        teacher_replay ? teacher.size() : max_tokens;
                    decoded.reserve(generation_limit);
                    generated_tokens.reserve(generation_limit);
                    token_ready_ms.reserve(generation_limit);
                    size_t body_calls = 1;
                    while (generated_tokens.size() < generation_limit &&
                           body_calls <= generation_limit + 100) {
                        speech_head.Run(hidden.hidden, logits);
                        const int32_t sampled =
                            embedded_rkllm::SampleWithEosRule(
                                logits, decoded,
                                static_cast<int>(decoded.size()) < min_tokens,
                                6561, generator);
                        int32_t chosen = sampled;
                        if (teacher_replay) {
                            chosen = teacher[generated_tokens.size()];
                        } else if (sampled == 6561) {
                            saw_eos = true;
                            stop_token = sampled;
                            break;
                        } else if (sampled > 6561) {
                            const float* repeated =
                                generated_tokens.empty()
                                    ? prefill.data() + prefill.size() - kHiddenSize
                                    : speech_embedding.data() +
                                          static_cast<size_t>(
                                              generated_tokens.back()) *
                                              kHiddenSize;
                            hidden = embedded_rkllm::RunHidden(
                                llm_handle, repeated, 1, true);
                            ++body_calls;
                            continue;
                        }
                        if (chosen < 0 || chosen >= 6561) {
                            throw std::runtime_error(
                                "LLM produced an invalid speech token");
                        }
                        decoded.push_back(chosen);
                        generated_tokens.push_back(chosen);
                        token_ready_ms.push_back(
                            Milliseconds(request_start, Clock::now()));
                        if (!token_queue.Push(chosen)) break;
                        if (generated_tokens.size() < generation_limit) {
                            hidden = embedded_rkllm::RunHidden(
                                llm_handle,
                                speech_embedding.data() +
                                    static_cast<size_t>(chosen) * kHiddenSize,
                                1, true);
                            ++body_calls;
                        }
                    }
                    llm_done_ms = Milliseconds(request_start, Clock::now());
                    token_queue.Close();
                } catch (...) {
                    record_error(std::current_exception());
                }
            });

            llm_thread.join();
            flow_thread.join();
            hift_thread.join();
            monitor_running.store(false);
            memory_monitor.join();
            if (worker_error != nullptr) std::rethrow_exception(worker_error);
            if (generated_tokens.empty() || produced_mel.empty() ||
                pcm_events.empty() ||
                waveform.size() != generated_tokens.size() * 2 *
                                       embedded_hift::kF0Upsample) {
                throw std::runtime_error("streaming pipeline produced incomplete output");
            }
            if (pcm_stream.Enabled() &&
                pcm_stream.BytesWritten() != waveform.size() * sizeof(int16_t)) {
                throw std::runtime_error("PCM stream byte count mismatch");
            }

            const auto target_mel = AssembleMel(produced_mel);
            const bool has_mel_reference =
                teacher_replay && target_mel.size() == expected_mel.size();
            const Metrics mel_metrics = has_mel_reference
                                            ? Compare(target_mel, expected_mel)
                                            : Metrics{};
            const auto raw_qc = embedded_hift::CheckAudio(waveform);
            if (!pcm_stream.Enabled() && raw_qc.peak > 0.99) {
                const float gain = static_cast<float>(0.99 / raw_qc.peak);
                for (float& sample : waveform) sample *= gain;
            }
            const fs::path wav_path = output_root / "streaming_segment00.wav";
            const fs::path mel_path = output_root / "streaming_segment00_mel.f32.bin";
            const fs::path token_path = output_root / "generated_tokens.i32.bin";
            if (!serve) {
                // 服务模式不落盘：一句一个文件只会留下上一句，而客户端要的
                // 是流出去的那一份。基准模式仍然照写，数字才可复现。
                embedded_hift::WritePcm16Wav(wav_path, waveform);
                WriteBinary(mel_path, target_mel);
                WriteBinary(token_path, generated_tokens);
            }

            size_t underruns = 0;
            double min_buffer_ms = std::numeric_limits<double>::infinity();
            for (const auto& event : pcm_events) {
                if (event.underrun) ++underruns;
                min_buffer_ms = std::min(min_buffer_ms, event.buffer_after_ms);
            }
            const double first_pcm_ms = pcm_events.front().ready_ms;
            const double first_pcm_seconds =
                static_cast<double>(pcm_events.front().playable_samples) /
                embedded_hift::kSampleRate;
            const double audio_seconds = generated_tokens.size() * 0.04;
            const double steady_audio_seconds = audio_seconds - first_pcm_seconds;
            const double steady_pcm_rtf =
                steady_audio_seconds > 0.0
                    ? (pcm_events.back().ready_ms - first_pcm_ms) /
                          (steady_audio_seconds * 1000.0)
                    : 0.0;

            if (!serve) {
                const fs::path flow_tsv_path = output_root / "flow_chunks.tsv";
                std::ofstream flow_tsv(flow_tsv_path, std::ios::trunc);
                flow_tsv << "global_chunk\tprompt\tcurrent_tokens\tcontext_tokens"
                            "\toutput_frames\tfinal\tstart_ms\tencoder_wait_ms"
                            "\tencoder_ms\tflow_wait_ms\tflow_ms"
                            "\tready_ms\n";
                for (const auto& event : flow_events) {
                    flow_tsv << event.global_chunk << '\t' << event.prompt << '\t'
                             << event.current_tokens << '\t' << event.context_tokens
                             << '\t' << event.output_frames << '\t' << event.final
                             << '\t' << event.start_ms << '\t'
                             << event.encoder_wait_ms << '\t' << event.encoder_ms
                             << '\t' << event.flow_wait_ms << '\t' << event.flow_ms
                             << '\t' << event.ready_ms << '\n';
                }
                const fs::path token_tsv_path = output_root / "llm_tokens.tsv";
                std::ofstream token_tsv(token_tsv_path, std::ios::trunc);
                token_tsv << "token_index\ttoken_id\tready_ms\tinterval_ms"
                             "\taudio_equivalent_ms\tinterval_rtf\n";
                for (size_t index = 0; index < generated_tokens.size(); ++index) {
                    const double interval =
                        index == 0 ? token_ready_ms[index]
                                   : token_ready_ms[index] - token_ready_ms[index - 1];
                    token_tsv << index << '\t' << generated_tokens[index] << '\t'
                              << token_ready_ms[index] << '\t' << interval << '\t'
                              << 40.0 << '\t' << interval / 40.0 << '\n';
                }
                const fs::path pcm_tsv_path = output_root / "pcm_chunks.tsv";
                std::ofstream pcm_tsv(pcm_tsv_path, std::ios::trunc);
                pcm_tsv << "call\tinput_frames\tmodel_frames\tplayable_samples"
                           "\tcumulative_samples"
                           "\tmel_ready_ms\tstart_ms\tready_ms\tf0_wait_ms\tf0_ms"
                           "\tdecoder_wait_ms\tdecoder_ms"
                           "\tcpu_ms\tstream_write_ms\tdeadline_margin_ms"
                           "\tbuffer_after_ms\tunderrun\n";
                for (const auto& event : pcm_events) {
                    pcm_tsv << event.call << '\t' << event.input_frames << '\t'
                            << event.model_frames << '\t' << event.playable_samples << '\t'
                            << event.cumulative_samples << '\t' << event.mel_ready_ms
                            << '\t' << event.start_ms << '\t' << event.ready_ms << '\t'
                            << event.f0_wait_ms << '\t' << event.f0_ms << '\t'
                            << event.decoder_wait_ms << '\t' << event.decoder_ms << '\t'
                            << event.cpu_ms << '\t' << event.stream_write_ms << '\t'
                            << event.deadline_margin_ms << '\t'
                            << event.buffer_after_ms << '\t' << event.underrun << '\n';
                }

                const fs::path metrics_path = output_root / "pipeline_metrics.json";
                WritePipelineJson(
                    metrics_path, teacher_replay, saw_eos, stop_token,
                    generated_tokens.size(), precompute_full_prompt,
                    steady_flow != nullptr, hift_batch2, serialize_downstream_npu,
                    dynamic_text,
                    planned_target_text_tokens.size(), prefill.size() / kHiddenSize,
                    frontend_asset_load_ms, request_frontend_ms, profile_ms,
                    init_warmup_ms,
                    token_ready_ms.empty() ? 0.0 : token_ready_ms.front(),
                    llm_done_ms, produced_mel.front().ready_ms, flow_done_ms,
                    first_pcm_ms, pipeline_done_ms, audio_seconds, steady_pcm_rtf,
                    underruns, min_buffer_ms, rss_peak.load(), token_queue.MaxSize(),
                    mel_queue.MaxSize(), has_mel_reference, pcm_stream_path,
                    pcm_stream.BytesWritten(), mel_metrics, flow_events, pcm_events,
                    wav_path, mel_path);
            }

            // 服务模式下 stdout 是纯 PCM，一个字节别的都不能有；报告改走 stderr，
            // 并用哨兵括起来，让 rkllm/rknn 自己打的噪声落在括号之外。
            std::ostream& report = serve ? std::cerr : std::cout;
            if (serve) report << "--- eidolon-tts utterance begin ---\n";
            report << std::setprecision(10)
                      << "status=PASS\n"
                      << "component=cosyvoice2_end_to_end_streaming_cpp\n"
                      << "teacher_replay=" << (teacher_replay ? 1 : 0) << '\n'
                      << "saw_eos=" << (saw_eos ? 1 : 0) << '\n'
                      << "stop_token=" << stop_token << '\n'
                      << "precompute_full_prompt="
                      << (precompute_full_prompt ? 1 : 0) << '\n'
                      << "hybrid_steady_flow="
                      << (steady_flow != nullptr ? 1 : 0) << '\n'
                      << "hift_batch2=" << (hift_batch2 ? 1 : 0) << '\n'
                      << "serialize_downstream_npu="
                      << (serialize_downstream_npu ? 1 : 0) << '\n'
                      << "text_frontend_mode="
                      << (dynamic_text ? "edge_dynamic" : "legacy_prefill_file")
                      << '\n'
                      << "target_text_tokens="
                      << planned_target_text_tokens.size() << '\n'
                      << "prefill_tokens=" << prefill.size() / kHiddenSize << '\n'
                      << "text_frontend_asset_load_ms="
                      << frontend_asset_load_ms << '\n'
                      << "request_text_frontend_ms=" << request_frontend_ms << '\n'
                      << "generated_tokens=" << generated_tokens.size() << '\n'
                      << "audio_seconds=" << audio_seconds << '\n'
                      << "service_init_warmup_ms=" << init_warmup_ms << '\n'
                      << "rkllm_init_ms=" << rkllm_init_ms << '\n'
                      << "profile_precompute_ms=" << profile_ms << '\n'
                      << "first_speech_token_ms=" << token_ready_ms.front() << '\n'
                      << "llm_done_ms=" << llm_done_ms << '\n'
                      << "first_target_mel_ms=" << produced_mel.front().ready_ms
                      << '\n'
                      << "flow_done_ms=" << flow_done_ms << '\n'
                      << "ttft_first_pcm_ms=" << first_pcm_ms << '\n'
                      << "pipeline_done_ms=" << pipeline_done_ms << '\n'
                      << "request_to_done_rtf="
                      << pipeline_done_ms / (audio_seconds * 1000.0) << '\n'
                      << "steady_pcm_rtf=" << steady_pcm_rtf << '\n'
                      << "underrun_count=" << underruns << '\n'
                      << "minimum_buffer_after_ms=" << min_buffer_ms << '\n'
                      << "token_queue_max=" << token_queue.MaxSize() << '\n'
                      << "mel_queue_max=" << mel_queue.MaxSize() << '\n'
                      << "pcm_stream_bytes=" << pcm_stream.BytesWritten() << '\n'
                      << "rss_peak_bytes=" << rss_peak.load() << '\n';
            if (!serve) {
                // 服务模式没写这些文件，报出它们的路径就是在说一件不真的事。
                report << "output_wav=" << wav_path.string() << '\n'
                       << "output_tokens=" << token_path.string() << '\n'
                       << "metrics_json="
                       << (output_root / "pipeline_metrics.json").string()
                       << '\n';
            }
            if (has_mel_reference) {
                report << "mel_cosine_vs_host_fixed=" << mel_metrics.cosine
                          << '\n'
                       << "mel_mae_vs_host_fixed=" << mel_metrics.mae << '\n'
                       << "mel_max_abs_vs_host_fixed=" << mel_metrics.max_abs
                       << '\n';
            }
            if (serve) {
                report << "--- eidolon-tts utterance end ---\n";
                report.flush();
            }
        };

        if (serve) {
            // 一行一句。文本里不会有换行，所以不需要在这里放一个 JSON 解析器；
            // 边界由行本身给出，而调用方是同一个仓库里的服务,不是任意客户端。
            //
            // 就绪信号必须在**加载和预热之后**发出：服务的调用者要等到它才开始
            // 派活，而在它之前发出的话，第一句会撞在还没预热完的图上。
            std::cerr << "--- eidolon-tts ready ---\n";
            std::cerr << "service_init_warmup_ms=" << init_warmup_ms << '\n';
            std::cerr.flush();
            std::string request_text;
            while (std::getline(std::cin, request_text)) {
                if (!request_text.empty() && request_text.back() == '\r') {
                    request_text.pop_back();
                }
                if (request_text.empty()) continue;
                try {
                    // 每句从空的 KV 开始。serve 模式复用同一个 RKLLM handle，而
                    // 每次 RunHidden 都传 keep_history=1——句内必须保持，否则续推
                    // 拿不到 prefill 留下的状态——所以上一句的历史会留在里面。
                    //
                    // 在 288 的上下文里它很快被挤出去，只表现为「同一句连说三遍
                    // token 数不同」这种无害抖动（§2.26 记过）。给它 2048 的余量
                    // 就会累积：第四句起模型开始复述音色 prompt 自己的文本，
                    // steady rtf 从 0.9 涨到 1.5（§2.27）。所以「清历史」和
                    // 「放大上下文」是同一件事的两半，必须一起做。
                    //
                    // 全清而不是清一个范围：范围形式要求 keep_history==0 且生成
                    // 已被回调返回 1 暂停（rkllm.h 的 @note），这里两个条件都不
                    // 成立。keep_history=0 也没有用在这里，它的语义是「这一次不
                    // 保留历史」，而句内的续推恰恰需要保留。
                    if (rkllm_clear_kv_cache(llm_handle, 0, nullptr, nullptr) != 0) {
                        throw std::runtime_error("could not clear the RKLLM KV cache");
                    }
                    // 自证，而不是相信。清完应当是 0；从音频上看这件事要到第四句
                    // 才看得出来，所以这一行是它唯一的即时证据。沉默即正常。
                    int kv_positions[8] = {0};
                    if (rkllm_get_kv_cache_size(llm_handle, kv_positions) == 0 &&
                        kv_positions[0] != 0) {
                        std::cerr << "kv_cache_not_empty_after_clear="
                                  << kv_positions[0] << '\n';
                        std::cerr.flush();
                    }
                    plan_text(request_text);
                    plan_mel_chunks();
                    synthesize(request_text);
                } catch (const std::exception& error) {
                    // 一句失败不该带走整个服务：模型还在内存里，下一句照常。
                    // 拒绝的理由要出去,否则调用方只看到一段没有音频的沉默。
                    std::cerr << "--- eidolon-tts utterance begin ---\n"
                              << "status=FAIL\n"
                              << "error=" << error.what() << '\n'
                              << "--- eidolon-tts utterance end ---\n";
                    std::cerr.flush();
                }
            }
        } else {
            synthesize(target_text);
        }

        rkllm_destroy(llm_handle);
        llm_handle = nullptr;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        if (llm_handle != nullptr) rkllm_destroy(llm_handle);
        return 1;
    }
}
