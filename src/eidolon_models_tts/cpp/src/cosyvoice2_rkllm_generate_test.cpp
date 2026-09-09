#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rkllm.h"
#include "rknn_api.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr size_t kHiddenSize = 896;
constexpr int32_t kDefaultSpeechVocabularySize = 6561;

double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void CheckRknn(int result, const std::string& operation) {
    if (result != RKNN_SUCC) {
        throw std::runtime_error(operation + " failed: " +
                                 std::to_string(result));
    }
}

template <typename T>
std::vector<T> ReadBinary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("failed to open " + path);
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary file " + path);
    }
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!values.empty()) {
        stream.read(reinterpret_cast<char*>(values.data()), bytes);
    }
    if (!stream) throw std::runtime_error("failed to read " + path);
    return values;
}

template <typename T>
void WriteBinary(const std::string& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to open " + path);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("failed to write " + path);
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
    if (exponent >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7c00u |
                                         (mantissa >> 13) | 1u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
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
            bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) |
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

void ValidateAffine(const rknn_tensor_attr& attr) {
    if (attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
        !std::isfinite(attr.scale) || attr.scale <= 0.0f) {
        throw std::runtime_error(std::string(attr.name) +
                                 " has invalid affine quantization metadata");
    }
}

struct Metrics {
    double cosine = 0.0;
    double mae = 0.0;
    double max_abs = 0.0;
};

Metrics Compare(const std::string& label, const std::vector<float>& actual,
                const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    double dot = 0.0;
    double norm_actual = 0.0;
    double norm_expected = 0.0;
    double absolute_error = 0.0;
    double max_error = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index];
        const double e = expected[index];
        const double error = std::abs(a - e);
        dot += a * e;
        norm_actual += a * a;
        norm_expected += e * e;
        absolute_error += error;
        max_error = std::max(max_error, error);
    }
    Metrics result;
    result.cosine = dot / std::sqrt(norm_actual * norm_expected);
    result.mae = absolute_error / actual.size();
    result.max_abs = max_error;
    std::cout << label << "_cosine=" << result.cosine << '\n'
              << label << "_mae=" << result.mae << '\n'
              << label << "_max_abs=" << result.max_abs << '\n';
    return result;
}

struct HiddenResult {
    std::array<float, kHiddenSize> hidden{};
    int input_tokens = 0;
    int returned_tokens = 0;
    double wall_ms = 0.0;
    bool error = false;
    bool has_hidden = false;
};

int UnsupportedTokenizer(void*, const char*, int32_t, int32_t*, int32_t) {
    return -1;
}

int UnsupportedEmbedding(void*, int32_t*, uint64_t, void*, uint64_t) {
    return -1;
}

int ResultCallback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* output = static_cast<HiddenResult*>(userdata);
    if (state == RKLLM_RUN_ERROR) {
        output->error = true;
        return 0;
    }
    if (state != RKLLM_RUN_NORMAL || result == nullptr) return 0;
    const auto& layer = result->last_hidden_layer;
    if (layer.hidden_states == nullptr || layer.num_tokens <= 0 ||
        layer.embd_size != static_cast<int>(kHiddenSize)) {
        return 0;
    }
    const float* last =
        layer.hidden_states +
        static_cast<size_t>(layer.num_tokens - 1) * layer.embd_size;
    std::copy(last, last + layer.embd_size, output->hidden.begin());
    output->returned_tokens = layer.num_tokens;
    output->has_hidden = true;
    return 2;
}

HiddenResult RunHidden(LLMHandle handle, const float* input, size_t tokens,
                       bool keep_history) {
    if (input == nullptr || tokens == 0) {
        throw std::runtime_error("invalid RKLLM embedding input");
    }
    RKLLMInput rkllm_input{};
    rkllm_input.input_type = RKLLM_INPUT_EMBED;
    rkllm_input.embed_input.embed = const_cast<float*>(input);
    rkllm_input.embed_input.n_tokens = tokens;
    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GET_LAST_HIDDEN_LAYER;
    infer.keep_history = keep_history ? 1 : 0;
    infer.max_new_tokens = 1;
    HiddenResult result;
    result.input_tokens = rkllm_input.embed_input.n_tokens;
    const auto start = Clock::now();
    const int status = rkllm_run(handle, &rkllm_input, &infer, &result);
    result.wall_ms = Milliseconds(start, Clock::now());
    if (status != 0 || result.error || !result.has_hidden) {
        throw std::runtime_error("rkllm_run failed: " +
                                 std::to_string(status));
    }
    return result;
}

struct HeadTiming {
    double bridge_ms = 0.0;
    double sync_ms = 0.0;
    double run_ms = 0.0;
    double readback_ms = 0.0;
};

class SpeechHead {
   public:
    SpeechHead(const std::string& model, rknn_core_mask core,
               size_t minimum_vocabulary_size) {
        CheckRknn(rknn_init(&context_, const_cast<char*>(model.c_str()), 0, 0,
                            nullptr),
                  "rknn_init(speech head)");
        CheckRknn(rknn_set_core_mask(context_, core), "rknn_set_core_mask");
        rknn_input_output_num count{};
        CheckRknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count,
                             sizeof(count)),
                  "RKNN_QUERY_IN_OUT_NUM");
        if (count.n_input != 1 || count.n_output != 1) {
            throw std::runtime_error("speech head must have one input/output");
        }
        input_attr_.index = 0;
        output_attr_.index = 0;
        CheckRknn(rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR,
                             &input_attr_, sizeof(input_attr_)),
                  "RKNN_QUERY_NATIVE_INPUT_ATTR");
        CheckRknn(rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                             &output_attr_, sizeof(output_attr_)),
                  "RKNN_QUERY_NATIVE_OUTPUT_ATTR");
        if ((input_attr_.type != RKNN_TENSOR_FLOAT16 &&
             input_attr_.type != RKNN_TENSOR_INT8) ||
            (output_attr_.type != RKNN_TENSOR_FLOAT16 &&
             output_attr_.type != RKNN_TENSOR_INT8) ||
            input_attr_.fmt != RKNN_TENSOR_UNDEFINED ||
            output_attr_.fmt != RKNN_TENSOR_UNDEFINED ||
            input_attr_.n_elems != kHiddenSize ||
            output_attr_.n_elems < minimum_vocabulary_size ||
            input_attr_.size_with_stride <
                input_attr_.n_elems *
                    (input_attr_.type == RKNN_TENSOR_FLOAT16 ? 2 : 1) ||
            output_attr_.size_with_stride <
                output_attr_.n_elems *
                    (output_attr_.type == RKNN_TENSOR_FLOAT16 ? 2 : 1)) {
            throw std::runtime_error("unexpected speech head native layout");
        }
        if (input_attr_.type == RKNN_TENSOR_INT8) ValidateAffine(input_attr_);
        if (output_attr_.type == RKNN_TENSOR_INT8) ValidateAffine(output_attr_);
        vocabulary_size_ = output_attr_.n_elems;
        input_ = rknn_create_mem(context_, input_attr_.size_with_stride);
        output_ = rknn_create_mem(context_, output_attr_.size_with_stride);
        if (input_ == nullptr || output_ == nullptr) {
            throw std::runtime_error("rknn_create_mem(speech head) failed");
        }
        std::memset(input_->virt_addr,
                    input_attr_.type == RKNN_TENSOR_INT8
                        ? static_cast<unsigned char>(
                              static_cast<int8_t>(input_attr_.zp))
                        : 0,
                    input_->size);
        std::memset(output_->virt_addr, 0, output_->size);
        CheckRknn(rknn_set_io_mem(context_, input_, &input_attr_),
                  "rknn_set_io_mem(speech head input)");
        CheckRknn(rknn_set_io_mem(context_, output_, &output_attr_),
                  "rknn_set_io_mem(speech head output)");
    }

    ~SpeechHead() {
        if (input_ != nullptr) rknn_destroy_mem(context_, input_);
        if (output_ != nullptr) rknn_destroy_mem(context_, output_);
        if (context_ != 0) rknn_destroy(context_);
    }

    void Warmup() {
        CheckRknn(rknn_mem_sync(context_, input_, RKNN_MEMORY_SYNC_TO_DEVICE),
                  "rknn_mem_sync(head warmup input)");
        CheckRknn(rknn_run(context_, nullptr), "rknn_run(head warmup)");
        CheckRknn(
            rknn_mem_sync(context_, output_, RKNN_MEMORY_SYNC_FROM_DEVICE),
            "rknn_mem_sync(head warmup output)");
    }

    HeadTiming Run(const std::array<float, kHiddenSize>& hidden,
                   std::vector<float>& logits) {
        logits.resize(vocabulary_size_);
        HeadTiming timing;
        const auto bridge_start = Clock::now();
        if (input_attr_.type == RKNN_TENSOR_FLOAT16) {
            auto* input_half = static_cast<uint16_t*>(input_->virt_addr);
            for (size_t index = 0; index < hidden.size(); ++index) {
                input_half[index] = FloatToHalf(hidden[index]);
            }
        } else {
            auto* input_int8 = static_cast<int8_t*>(input_->virt_addr);
            for (size_t index = 0; index < hidden.size(); ++index) {
                const double quantized =
                    std::nearbyint(static_cast<double>(hidden[index]) /
                                   input_attr_.scale) +
                    input_attr_.zp;
                input_int8[index] = static_cast<int8_t>(std::clamp(
                    quantized,
                    static_cast<double>(
                        std::numeric_limits<int8_t>::lowest()),
                    static_cast<double>(
                        std::numeric_limits<int8_t>::max())));
            }
        }
        const auto bridge_end = Clock::now();
        CheckRknn(rknn_mem_sync(context_, input_, RKNN_MEMORY_SYNC_TO_DEVICE),
                  "rknn_mem_sync(head input)");
        const auto input_sync_end = Clock::now();
        CheckRknn(rknn_run(context_, nullptr), "rknn_run(speech head)");
        const auto run_end = Clock::now();
        CheckRknn(
            rknn_mem_sync(context_, output_, RKNN_MEMORY_SYNC_FROM_DEVICE),
            "rknn_mem_sync(head output)");
        const auto output_sync_end = Clock::now();
        if (output_attr_.type == RKNN_TENSOR_FLOAT16) {
            const auto* output_half =
                static_cast<const uint16_t*>(output_->virt_addr);
            for (size_t index = 0; index < logits.size(); ++index) {
                logits[index] = HalfToFloat(output_half[index]);
            }
        } else {
            const auto* output_int8 =
                static_cast<const int8_t*>(output_->virt_addr);
            for (size_t index = 0; index < logits.size(); ++index) {
                logits[index] =
                    (static_cast<int32_t>(output_int8[index]) -
                     output_attr_.zp) *
                    output_attr_.scale;
            }
        }
        const auto readback_end = Clock::now();
        timing.bridge_ms = Milliseconds(bridge_start, bridge_end);
        timing.sync_ms = Milliseconds(bridge_end, input_sync_end) +
                         Milliseconds(run_end, output_sync_end);
        timing.run_ms = Milliseconds(input_sync_end, run_end);
        timing.readback_ms = Milliseconds(output_sync_end, readback_end);
        return timing;
    }

    size_t VocabularySize() const { return vocabulary_size_; }

   private:
    rknn_context context_ = 0;
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr output_attr_{};
    rknn_tensor_mem* input_ = nullptr;
    rknn_tensor_mem* output_ = nullptr;
    size_t vocabulary_size_ = 0;
};

std::vector<double> Softmax(const std::vector<float>& logits) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size());
    double total = 0.0;
    for (size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(static_cast<double>(logits[index] - maximum));
        total += probabilities[index];
    }
    for (double& value : probabilities) value /= total;
    return probabilities;
}

int32_t WeightedSample(const std::vector<int32_t>& ids,
                       const std::vector<double>& weights,
                       std::mt19937& generator) {
    std::discrete_distribution<size_t> distribution(weights.begin(),
                                                    weights.end());
    return ids[distribution(generator)];
}

int32_t WeightedSampleAll(const std::vector<double>& weights,
                          std::mt19937& generator) {
    std::discrete_distribution<size_t> distribution(weights.begin(),
                                                    weights.end());
    return static_cast<int32_t>(distribution(generator));
}

int32_t RasSample(const std::vector<float>& logits,
                  const std::vector<int32_t>& decoded,
                  std::mt19937& generator) {
    const std::vector<double> probabilities = Softmax(logits);
    std::vector<int32_t> top_ids(probabilities.size());
    std::iota(top_ids.begin(), top_ids.end(), 0);
    const size_t top_count = std::min<size_t>(25, top_ids.size());
    std::partial_sort(
        top_ids.begin(), top_ids.begin() + static_cast<ptrdiff_t>(top_count),
        top_ids.end(), [&](int32_t left, int32_t right) {
            if (probabilities[left] != probabilities[right]) {
                return probabilities[left] > probabilities[right];
            }
            return left < right;
        });
    std::vector<int32_t> nucleus_ids;
    std::vector<double> nucleus_weights;
    nucleus_ids.reserve(top_count);
    nucleus_weights.reserve(top_count);
    double cumulative = 0.0;
    for (size_t index = 0; index < top_count && cumulative < 0.8; ++index) {
        const int32_t id = top_ids[index];
        cumulative += probabilities[id];
        nucleus_ids.push_back(id);
        nucleus_weights.push_back(probabilities[id]);
    }
    int32_t token = WeightedSample(nucleus_ids, nucleus_weights, generator);
    const size_t begin = decoded.size() > 10 ? decoded.size() - 10 : 0;
    const size_t repetitions = static_cast<size_t>(std::count(
        decoded.begin() + static_cast<ptrdiff_t>(begin), decoded.end(), token));
    if (repetitions >= 1) {
        token = WeightedSampleAll(probabilities, generator);
    }
    return token;
}

int32_t SampleWithEosRule(const std::vector<float>& logits,
                          const std::vector<int32_t>& decoded,
                          bool ignore_eos, int32_t eos_token,
                          std::mt19937& generator) {
    for (int trial = 0; trial <= 100; ++trial) {
        const int32_t token = RasSample(logits, decoded, generator);
        if (!ignore_eos || token != eos_token) return token;
    }
    throw std::runtime_error("sampling exceeded EOS retry limit");
}

rknn_core_mask ParseCore(const std::string& value) {
    if (value == "auto") return RKNN_NPU_CORE_AUTO;
    if (value == "0") return RKNN_NPU_CORE_0;
    if (value == "1") return RKNN_NPU_CORE_1;
    if (value == "2") return RKNN_NPU_CORE_2;
    if (value == "01") return RKNN_NPU_CORE_0_1;
    if (value == "012") return RKNN_NPU_CORE_0_1_2;
    if (value == "all") return RKNN_NPU_CORE_ALL;
    throw std::runtime_error("invalid --core value: " + value);
}

double Mean(const std::vector<double>& values) {
    return values.empty()
               ? 0.0
               : std::accumulate(values.begin(), values.end(), 0.0) /
                     values.size();
}

double Percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

struct GenerationResult {
    std::vector<int32_t> generated;
    std::vector<int32_t> sampled;
    std::vector<double> token_ready_ms;
    size_t prefill_tokens = 0;
    size_t expected_tokens = 0;
    size_t expected_prefix_match = 0;
    size_t expected_total_match = 0;
    size_t teacher_sample_matches = 0;
    size_t invalid_control_tokens = 0;
    int body_calls = 0;
    int decode_calls = 0;
    int32_t stop_token = -1;
    bool saw_eos = false;
    bool saw_stop = false;
    uint64_t hidden_trace_hash = 1469598103934665603ull;
    double prefill_ms = 0.0;
    double decode_total_ms = 0.0;
    double decode_mean_ms = 0.0;
    double head_bridge_ms = 0.0;
    double head_sync_ms = 0.0;
    double head_npu_ms = 0.0;
    double head_readback_ms = 0.0;
    double head_total_ms = 0.0;
    double sampling_ms = 0.0;
    double generation_ms = 0.0;
    double first_token_ms = 0.0;
    double token_interval_mean_ms = 0.0;
    double token_interval_p50_ms = 0.0;
    double token_interval_p90_ms = 0.0;
    double token_interval_max_ms = 0.0;
    double steady_token_rtf = 0.0;
};

void HashHidden(uint64_t& hash,
                const std::array<float, kHiddenSize>& hidden) {
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    for (float value : hidden) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(bits >> shift);
            hash *= kFnvPrime;
        }
    }
}

GenerationResult GenerateCase(
    LLMHandle handle, SpeechHead& speech_head,
    const std::vector<float>& speech_embedding,
    const std::vector<float>& prefill, const std::vector<int32_t>& expected,
    const std::vector<int32_t>& teacher,
    const std::vector<float>* prefill_expected, int min_tokens, int max_tokens,
    int32_t speech_vocabulary_size, uint32_t seed) {
    if (prefill.empty() || prefill.size() % kHiddenSize != 0) {
        throw std::runtime_error("invalid RKLLM prefill tensor");
    }
    if (!teacher.empty()) max_tokens = static_cast<int>(teacher.size());

    const auto case_start = Clock::now();
    GenerationResult result;
    result.prefill_tokens = prefill.size() / kHiddenSize;
    result.expected_tokens = expected.size();
    HiddenResult hidden =
        RunHidden(handle, prefill.data(), result.prefill_tokens, true);
    result.prefill_ms = hidden.wall_ms;
    HashHidden(result.hidden_trace_hash, hidden.hidden);
    if (prefill_expected != nullptr) {
        if (prefill_expected->size() < kHiddenSize) {
            throw std::runtime_error("prefill expected tensor is too small");
        }
        const std::vector<float> actual(hidden.hidden.begin(),
                                        hidden.hidden.end());
        const std::vector<float> last_expected(
            prefill_expected->end() - static_cast<ptrdiff_t>(kHiddenSize),
            prefill_expected->end());
        Compare("prefill_last_hidden", actual, last_expected);
    }

    std::mt19937 generator(seed);
    result.generated.reserve(static_cast<size_t>(max_tokens));
    result.sampled.reserve(static_cast<size_t>(max_tokens));
    std::vector<float> logits;
    std::vector<double> decode_times;
    decode_times.reserve(static_cast<size_t>(max_tokens));
    result.body_calls = 1;

    while (static_cast<int>(result.generated.size()) < max_tokens &&
           result.body_calls <= max_tokens + 100) {
        const HeadTiming head_timing = speech_head.Run(hidden.hidden, logits);
        result.head_bridge_ms += head_timing.bridge_ms;
        result.head_sync_ms += head_timing.sync_ms;
        result.head_npu_ms += head_timing.run_ms;
        result.head_readback_ms += head_timing.readback_ms;
        const auto sampling_start = Clock::now();
        const bool ignore_eos =
            static_cast<int>(result.generated.size()) < min_tokens;
        const int32_t sampled = SampleWithEosRule(
            logits, result.generated, ignore_eos, speech_vocabulary_size,
            generator);
        result.sampled.push_back(sampled);
        result.sampling_ms += Milliseconds(sampling_start, Clock::now());

        int32_t chosen = sampled;
        if (!teacher.empty()) {
            chosen = teacher[result.generated.size()];
            if (sampled == chosen) ++result.teacher_sample_matches;
        }
        if (chosen >= speech_vocabulary_size) {
            result.saw_eos = chosen == speech_vocabulary_size;
            result.saw_stop = true;
            result.stop_token = chosen;
            break;
        }
        if (chosen < 0) {
            throw std::runtime_error("sampled token is out of range");
        }
        result.generated.push_back(chosen);
        result.token_ready_ms.push_back(
            Milliseconds(case_start, Clock::now()));
        if (!teacher.empty() && result.generated.size() == teacher.size()) {
            break;
        }

        const size_t offset = static_cast<size_t>(chosen) * kHiddenSize;
        hidden = RunHidden(handle, speech_embedding.data() + offset, 1, true);
        HashHidden(result.hidden_trace_hash, hidden.hidden);
        decode_times.push_back(hidden.wall_ms);
        ++result.body_calls;
    }

    const size_t common = std::min(expected.size(), result.generated.size());
    while (result.expected_prefix_match < common &&
           expected[result.expected_prefix_match] ==
               result.generated[result.expected_prefix_match]) {
        ++result.expected_prefix_match;
    }
    for (size_t index = 0; index < common; ++index) {
        if (expected[index] == result.generated[index]) {
            ++result.expected_total_match;
        }
    }
    result.decode_calls = static_cast<int>(decode_times.size());
    result.decode_total_ms =
        std::accumulate(decode_times.begin(), decode_times.end(), 0.0);
    result.decode_mean_ms = Mean(decode_times);
    result.head_total_ms = result.head_bridge_ms + result.head_sync_ms +
                           result.head_npu_ms + result.head_readback_ms;
    result.generation_ms = result.prefill_ms + result.decode_total_ms +
                           result.head_total_ms + result.sampling_ms;
    if (!result.token_ready_ms.empty()) {
        result.first_token_ms = result.token_ready_ms.front();
    }
    std::vector<double> token_intervals;
    token_intervals.reserve(result.token_ready_ms.size());
    for (size_t index = 1; index < result.token_ready_ms.size(); ++index) {
        token_intervals.push_back(result.token_ready_ms[index] -
                                  result.token_ready_ms[index - 1]);
    }
    result.token_interval_mean_ms = Mean(token_intervals);
    result.token_interval_p50_ms = Percentile(token_intervals, 0.5);
    result.token_interval_p90_ms = Percentile(token_intervals, 0.9);
    result.token_interval_max_ms =
        token_intervals.empty()
            ? 0.0
            : *std::max_element(token_intervals.begin(),
                                token_intervals.end());
    if (!token_intervals.empty()) {
        constexpr double kAudioMsPerToken = 40.0;
        result.steady_token_rtf =
            std::accumulate(token_intervals.begin(), token_intervals.end(),
                            0.0) /
            (token_intervals.size() * kAudioMsPerToken);
    }
    return result;
}

LLMHandle InitializeRkllm(const std::string& model, int max_context,
                          int enabled_cpus_num, uint32_t enabled_cpus_mask,
                          double& elapsed_ms) {
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model.c_str();
    param.max_context_len = max_context;
    param.max_new_tokens = 1;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 0;
    param.extend_param.enabled_cpus_num = enabled_cpus_num;
    param.extend_param.enabled_cpus_mask = enabled_cpus_mask;
    RKLLMCallback callback{};
    callback.result_callback = ResultCallback;
    callback.result_userdata = nullptr;
    callback.tokenizer_callback = UnsupportedTokenizer;
    callback.embed_callback = UnsupportedEmbedding;

    LLMHandle handle = nullptr;
    const auto start = Clock::now();
    const int status = rkllm_init(&handle, &param, &callback);
    elapsed_ms = Milliseconds(start, Clock::now());
    if (status != 0 || handle == nullptr) {
        throw std::runtime_error("rkllm_init failed: " +
                                 std::to_string(status));
    }
    return handle;
}

void PrintGeneration(const GenerationResult& result, const std::string& mode,
                     const std::string& model_name, uint32_t seed,
                     const std::string& core_label,
                     int32_t speech_vocabulary_size, size_t head_vocabulary,
                     double init_ms, double rkllm_init_ms,
                     double speech_head_init_ms, double speech_head_warmup_ms,
                     int max_context, int enabled_cpus_num,
                     uint32_t enabled_cpus_mask, const std::string& output_path) {
    std::cout << std::setprecision(9)
              << "component=" << model_name << "_rkllm_speech_generation\n"
              << "mode=" << mode << '\n'
              << "seed=" << seed << '\n'
              << "npu_core=" << core_label << '\n'
              << "max_context=" << max_context << '\n'
              << "enabled_cpus_num=" << enabled_cpus_num << '\n'
              << "enabled_cpus_mask=" << enabled_cpus_mask << '\n'
              << "prefill_tokens=" << result.prefill_tokens << '\n'
              << "generated_tokens=" << result.generated.size() << '\n'
              << "head_vocabulary=" << head_vocabulary << '\n'
              << "speech_vocabulary=" << speech_vocabulary_size << '\n'
              << "saw_eos=" << (result.saw_eos ? 1 : 0) << '\n'
              << "saw_stop=" << (result.saw_stop ? 1 : 0) << '\n'
              << "stop_token=" << result.stop_token << '\n'
              << "invalid_control_tokens=" << result.invalid_control_tokens
              << '\n'
              << "body_calls=" << result.body_calls << '\n'
              << "init_ms=" << init_ms << '\n'
              << "rkllm_init_ms=" << rkllm_init_ms << '\n'
              << "speech_head_init_ms=" << speech_head_init_ms << '\n'
              << "speech_head_warmup_ms=" << speech_head_warmup_ms << '\n'
              << "prefill_ms=" << result.prefill_ms << '\n'
              << "decode_calls=" << result.decode_calls << '\n'
              << "decode_total_ms=" << result.decode_total_ms << '\n'
              << "decode_mean_ms=" << result.decode_mean_ms << '\n'
              << "head_bridge_ms=" << result.head_bridge_ms << '\n'
              << "head_sync_ms=" << result.head_sync_ms << '\n'
              << "head_npu_ms=" << result.head_npu_ms << '\n'
              << "head_readback_ms=" << result.head_readback_ms << '\n'
              << "head_total_ms=" << result.head_total_ms << '\n'
              << "sampling_ms=" << result.sampling_ms << '\n'
              << "generation_ms=" << result.generation_ms << '\n'
              << "first_speech_token_ms=" << result.first_token_ms << '\n'
              << "token_interval_mean_ms="
              << result.token_interval_mean_ms << '\n'
              << "token_interval_p50_ms=" << result.token_interval_p50_ms
              << '\n'
              << "token_interval_p90_ms=" << result.token_interval_p90_ms
              << '\n'
              << "token_interval_max_ms=" << result.token_interval_max_ms
              << '\n'
              << "steady_token_rtf=" << result.steady_token_rtf << '\n'
              << "output_tokens_per_second="
              << (result.generated.empty()
                      ? 0.0
                      : 1000.0 * result.generated.size() /
                            result.generation_ms)
              << '\n'
              << "rkllm_history=1\n"
              << "speech_head_native_io_mem=1\n"
              << "embedding_lookup_zero_copy=1\n"
              << "expected_tokens=" << result.expected_tokens << '\n'
              << "expected_prefix_match=" << result.expected_prefix_match
              << '\n'
              << "expected_total_match=" << result.expected_total_match << '\n'
              << "teacher_sample_matches=" << result.teacher_sample_matches
              << '\n'
              << "hidden_trace_hash=" << result.hidden_trace_hash << '\n'
              << "output_path=" << output_path << '\n';
}

struct BatchRow {
    std::string case_id;
    uint32_t seed = 0;
    size_t generated_tokens = 0;
    size_t target_samples = 0;
};

std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    if (!line.empty() && line.back() == '\t') fields.emplace_back();
    return fields;
}

std::vector<BatchRow> ReadBatchManifest(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::string line;
    if (!std::getline(stream, line)) {
        throw std::runtime_error("empty manifest " + path.string());
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto names = SplitTabs(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t index = 0; index < names.size(); ++index) {
        columns.emplace(names[index], index);
    }
    for (const char* required : {"case_id", "seed", "generated_tokens",
                                 "target_samples"}) {
        if (columns.count(required) == 0) {
            throw std::runtime_error("manifest is missing column " +
                                     std::string(required));
        }
    }

    std::vector<BatchRow> rows;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = SplitTabs(line);
        const auto get = [&](const char* name) -> const std::string& {
            const size_t index = columns.at(name);
            if (index >= fields.size()) {
                throw std::runtime_error("short manifest row: " + line);
            }
            return fields[index];
        };
        BatchRow row;
        row.case_id = get("case_id");
        row.seed = static_cast<uint32_t>(std::stoul(get("seed")));
        row.generated_tokens = std::stoull(get("generated_tokens"));
        row.target_samples = std::stoull(get("target_samples"));
        if (row.case_id.empty() || row.generated_tokens == 0 ||
            row.target_samples == 0) {
            throw std::runtime_error("invalid manifest row: " + line);
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        throw std::runtime_error("empty manifest " + path.string());
    }
    return rows;
}

int BatchMain(int argc, char** argv) {
    if (argc < 8) {
        std::cerr
            << "usage: cosyvoice2_rkllm_generate_test --batch RKLLM_MODEL "
               "HEAD_RKNN SPEECH_EMBEDDING_F32 BENCHMARK_ROOT MANIFEST_TSV "
               "OUTPUT_ROOT [--core=auto|0|1|2|01|012|all] "
               "[--speech-vocab=N] [--min-tokens=N] [--max-tokens=N] "
               "[--max-context=N] [--cpu-count=N] [--cpu-mask=N]\n";
        return 2;
    }

    LLMHandle handle = nullptr;
    try {
        const std::string rkllm_model = argv[2];
        const std::string head_model = argv[3];
        const std::string embedding_path = argv[4];
        const fs::path benchmark_root = argv[5];
        const fs::path manifest_path = argv[6];
        const fs::path output_root = argv[7];
        int min_tokens = 74;
        int max_tokens = 740;
        int max_context = 2048;
        int enabled_cpus_num = 4;
        uint32_t enabled_cpus_mask = 0xF0;
        int32_t speech_vocabulary_size = kDefaultSpeechVocabularySize;
        rknn_core_mask core = RKNN_NPU_CORE_0_1_2;
        std::string core_label = "012";
        for (int index = 8; index < argc; ++index) {
            const std::string option = argv[index];
            if (option.rfind("--core=", 0) == 0) {
                core_label = option.substr(7);
                core = ParseCore(core_label);
            } else if (option.rfind("--speech-vocab=", 0) == 0) {
                speech_vocabulary_size = std::stoi(option.substr(15));
            } else if (option.rfind("--min-tokens=", 0) == 0) {
                min_tokens = std::stoi(option.substr(13));
            } else if (option.rfind("--max-tokens=", 0) == 0) {
                max_tokens = std::stoi(option.substr(13));
            } else if (option.rfind("--max-context=", 0) == 0) {
                max_context = std::stoi(option.substr(14));
            } else if (option.rfind("--cpu-count=", 0) == 0) {
                enabled_cpus_num = std::stoi(option.substr(12));
            } else if (option.rfind("--cpu-mask=", 0) == 0) {
                enabled_cpus_mask = static_cast<uint32_t>(
                    std::stoul(option.substr(11), nullptr, 0));
            } else {
                throw std::runtime_error("unknown option: " + option);
            }
        }
        if (min_tokens < 0 || max_tokens <= min_tokens || max_context <= 0 ||
            enabled_cpus_num <= 0 || enabled_cpus_num > 8 ||
            enabled_cpus_mask == 0 || speech_vocabulary_size <= 0) {
            throw std::runtime_error("invalid batch options");
        }

        const auto rows = ReadBatchManifest(manifest_path);
        const auto speech_embedding = ReadBinary<float>(embedding_path);
        if (speech_embedding.empty() ||
            speech_embedding.size() % kHiddenSize != 0) {
            throw std::runtime_error("invalid speech embedding tensor");
        }
        const size_t embedding_vocabulary_size =
            speech_embedding.size() / kHiddenSize;
        if (embedding_vocabulary_size <
            static_cast<size_t>(speech_vocabulary_size)) {
            throw std::runtime_error("speech embedding vocabulary is too small");
        }
        fs::create_directories(output_root);

        const auto init_start = Clock::now();
        double rkllm_init_ms = 0.0;
        handle = InitializeRkllm(rkllm_model, max_context, enabled_cpus_num,
                                 enabled_cpus_mask, rkllm_init_ms);
        const auto head_init_start = Clock::now();
        SpeechHead speech_head(head_model, core, speech_vocabulary_size);
        const double speech_head_init_ms =
            Milliseconds(head_init_start, Clock::now());
        if (speech_head.VocabularySize() != embedding_vocabulary_size) {
            throw std::runtime_error(
                "speech head and embedding vocabulary sizes differ");
        }
        const auto warmup_start = Clock::now();
        speech_head.Warmup();
        const double speech_head_warmup_ms =
            Milliseconds(warmup_start, Clock::now());
        const double init_ms = Milliseconds(init_start, Clock::now());

        const fs::path tsv_path = output_root / "rkllm_batch.tsv";
        std::ofstream tsv(tsv_path, std::ios::trunc);
        if (!tsv) {
            throw std::runtime_error("failed to create " + tsv_path.string());
        }
        const fs::path token_tsv_path =
            output_root / "rkllm_token_timing.tsv";
        std::ofstream token_tsv(token_tsv_path, std::ios::trunc);
        if (!token_tsv) {
            throw std::runtime_error("failed to create " +
                                     token_tsv_path.string());
        }
        tsv << "case_id\tseed\tprefill_tokens\tgenerated_tokens"
               "\ttarget_samples\taudio_seconds\tkv_clear_ms\tprefill_ms"
               "\tdecode_calls\tdecode_total_ms\thead_total_ms\tsampling_ms"
               "\tgeneration_ms\tcase_wall_ms\tfirst_speech_token_ms"
               "\ttoken_interval_mean_ms\ttoken_interval_p50_ms"
               "\ttoken_interval_p90_ms\ttoken_interval_max_ms"
               "\tsteady_token_rtf"
               "\thidden_trace_hash\texact_teacher_match\n";
        token_tsv << "case_id\ttoken_index\tready_ms\tinterval_ms"
                     "\taudio_equivalent_ms\tinterval_rtf\n";

        size_t total_tokens = 0;
        size_t exact_cases = 0;
        size_t total_prefill_tokens = 0;
        double audio_seconds = 0.0;
        double clear_total_ms = 0.0;
        double prefill_total_ms = 0.0;
        double decode_total_ms = 0.0;
        double head_total_ms = 0.0;
        double sampling_total_ms = 0.0;
        double generation_total_ms = 0.0;
        const auto batch_start = Clock::now();
        for (size_t index = 0; index < rows.size(); ++index) {
            const BatchRow& row = rows[index];
            const auto case_start = Clock::now();
            double clear_ms = 0.0;
            if (index != 0) {
                const auto clear_start = Clock::now();
                const int status =
                    rkllm_clear_kv_cache(handle, 0, nullptr, nullptr);
                clear_ms = Milliseconds(clear_start, Clock::now());
                if (status != 0) {
                    throw std::runtime_error("rkllm_clear_kv_cache failed: " +
                                             std::to_string(status));
                }
            }

            const fs::path runtime =
                benchmark_root / "cases" / row.case_id / "runtime_input";
            const auto prefill =
                ReadBinary<float>((runtime / "rkllm_prefill.f32.bin").string());
            const auto teacher = ReadBinary<int32_t>(
                (runtime / "generated_token.i32.bin").string());
            if (teacher.size() != row.generated_tokens) {
                throw std::runtime_error("teacher token count mismatch for " +
                                         row.case_id);
            }
            const GenerationResult generated = GenerateCase(
                handle, speech_head, speech_embedding, prefill, teacher,
                teacher, nullptr, min_tokens, max_tokens,
                speech_vocabulary_size, row.seed);
            const fs::path output_path = output_root / (row.case_id + ".i32.bin");
            WriteBinary(output_path.string(), generated.generated);
            WriteBinary((output_root / (row.case_id + ".sampled.i32.bin")).string(),
                        generated.sampled);
            const bool exact = generated.generated == teacher;
            if (!exact) {
                throw std::runtime_error("teacher replay mismatch for " +
                                         row.case_id);
            }
            const double case_wall_ms =
                Milliseconds(case_start, Clock::now());
            const double case_audio_seconds =
                static_cast<double>(row.target_samples) / 24000.0;
            tsv << std::setprecision(10) << row.case_id << '\t' << row.seed
                << '\t' << generated.prefill_tokens << '\t'
                << generated.generated.size() << '\t' << row.target_samples
                << '\t' << case_audio_seconds << '\t' << clear_ms << '\t'
                << generated.prefill_ms << '\t' << generated.decode_calls
                << '\t' << generated.decode_total_ms << '\t'
                << generated.head_total_ms << '\t' << generated.sampling_ms
                << '\t' << generated.generation_ms << '\t' << case_wall_ms
                << '\t' << generated.first_token_ms << '\t'
                << generated.token_interval_mean_ms << '\t'
                << generated.token_interval_p50_ms << '\t'
                << generated.token_interval_p90_ms << '\t'
                << generated.token_interval_max_ms << '\t'
                << generated.steady_token_rtf << '\t'
                << generated.hidden_trace_hash << "\t1\n";
            tsv.flush();
            for (size_t token_index = 0;
                 token_index < generated.token_ready_ms.size();
                 ++token_index) {
                const double interval_ms =
                    token_index == 0
                        ? generated.token_ready_ms[token_index]
                        : generated.token_ready_ms[token_index] -
                              generated.token_ready_ms[token_index - 1];
                token_tsv << std::setprecision(10) << row.case_id << '\t'
                          << token_index << '\t'
                          << generated.token_ready_ms[token_index] << '\t'
                          << interval_ms << '\t' << 40.0 << '\t'
                          << interval_ms / 40.0 << '\n';
            }
            token_tsv.flush();

            ++exact_cases;
            total_tokens += generated.generated.size();
            total_prefill_tokens += generated.prefill_tokens;
            audio_seconds += case_audio_seconds;
            clear_total_ms += clear_ms;
            prefill_total_ms += generated.prefill_ms;
            decode_total_ms += generated.decode_total_ms;
            head_total_ms += generated.head_total_ms;
            sampling_total_ms += generated.sampling_ms;
            generation_total_ms += generated.generation_ms;
            std::cout << std::setprecision(9) << "case=" << row.case_id
                      << " generated_tokens=" << generated.generated.size()
                      << " kv_clear_ms=" << clear_ms
                      << " prefill_ms=" << generated.prefill_ms
                      << " decode_ms=" << generated.decode_total_ms
                      << " generation_ms=" << generated.generation_ms
                      << " hidden_trace_hash=" << generated.hidden_trace_hash
                      << " exact_teacher_match=1\n";
        }
        const double batch_wall_ms = Milliseconds(batch_start, Clock::now());
        const double end_to_end_ms = Milliseconds(init_start, Clock::now());
        std::cout << std::setprecision(10)
                  << "status=PASS\n"
                  << "component=cosyvoice2_rkllm_persistent_batch\n"
                  << "cases=" << rows.size() << '\n'
                  << "exact_teacher_cases=" << exact_cases << '\n'
                  << "generated_tokens_total=" << total_tokens << '\n'
                  << "prefill_tokens_total=" << total_prefill_tokens << '\n'
                  << "audio_seconds_total=" << audio_seconds << '\n'
                  << "npu_core=" << core_label << '\n'
                  << "max_context=" << max_context << '\n'
                  << "enabled_cpus_num=" << enabled_cpus_num << '\n'
                  << "enabled_cpus_mask=" << enabled_cpus_mask << '\n'
                  << "rkllm_init_once=1\n"
                  << "rkllm_kv_clear_between_cases=1\n"
                  << "speech_head_native_io_mem=1\n"
                  << "embedding_lookup_zero_copy=1\n"
                  << "init_ms=" << init_ms << '\n'
                  << "rkllm_init_ms=" << rkllm_init_ms << '\n'
                  << "speech_head_init_ms=" << speech_head_init_ms << '\n'
                  << "speech_head_warmup_ms=" << speech_head_warmup_ms << '\n'
                  << "kv_clear_total_ms=" << clear_total_ms << '\n'
                  << "prefill_total_ms=" << prefill_total_ms << '\n'
                  << "decode_total_ms=" << decode_total_ms << '\n'
                  << "head_total_ms=" << head_total_ms << '\n'
                  << "sampling_total_ms=" << sampling_total_ms << '\n'
                  << "generation_total_ms=" << generation_total_ms << '\n'
                  << "batch_wall_ms=" << batch_wall_ms << '\n'
                  << "end_to_end_ms=" << end_to_end_ms << '\n'
                  << "generation_rtf="
                  << generation_total_ms / (audio_seconds * 1000.0) << '\n'
                  << "batch_wall_rtf="
                  << batch_wall_ms / (audio_seconds * 1000.0) << '\n'
                  << "end_to_end_rtf="
                  << end_to_end_ms / (audio_seconds * 1000.0) << '\n'
                  << "output_tokens_per_second="
                  << 1000.0 * total_tokens / generation_total_ms << '\n'
                  << "result_tsv=" << tsv_path.string() << '\n'
                  << "token_result_tsv=" << token_tsv_path.string() << '\n';
        rkllm_destroy(handle);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        if (handle != nullptr) rkllm_destroy(handle);
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--batch") {
        return BatchMain(argc, argv);
    }
    if (argc < 6) {
        std::cerr
            << "usage: cosyvoice2_rkllm_generate_test RKLLM_MODEL HEAD_RKNN "
               "SPEECH_EMBEDDING_F32 PREFILL_F32 OUTPUT_TOKEN_I32 "
               "[--expected-tokens=PATH] [--teacher-tokens=PATH] "
               "[--prefill-expected=PATH] [--min-tokens=N] "
               "[--max-tokens=N] [--seed=N] [--core=auto|0|1|2|01|012|all] "
               "[--speech-vocab=N] [--model-name=NAME] [--max-context=N] "
               "[--cpu-count=N] [--cpu-mask=N]\n";
        return 2;
    }
    LLMHandle handle = nullptr;
    try {
        const std::string rkllm_model = argv[1];
        const std::string head_model = argv[2];
        const std::string embedding_path = argv[3];
        const std::string prefill_path = argv[4];
        const std::string output_path = argv[5];
        std::string expected_path;
        std::string teacher_path;
        std::string prefill_expected_path;
        int min_tokens = 74;
        int max_tokens = 740;
        int max_context = 2048;
        int enabled_cpus_num = 4;
        uint32_t enabled_cpus_mask = 0xF0;
        int32_t speech_vocabulary_size = kDefaultSpeechVocabularySize;
        std::string model_name = "cosyvoice2";
        uint32_t seed = 1986;
        rknn_core_mask core = RKNN_NPU_CORE_2;
        std::string core_label = "2";
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option.rfind("--expected-tokens=", 0) == 0) {
                expected_path = option.substr(18);
            } else if (option.rfind("--teacher-tokens=", 0) == 0) {
                teacher_path = option.substr(17);
            } else if (option.rfind("--prefill-expected=", 0) == 0) {
                prefill_expected_path = option.substr(19);
            } else if (option.rfind("--min-tokens=", 0) == 0) {
                min_tokens = std::stoi(option.substr(13));
            } else if (option.rfind("--max-tokens=", 0) == 0) {
                max_tokens = std::stoi(option.substr(13));
            } else if (option.rfind("--seed=", 0) == 0) {
                seed = static_cast<uint32_t>(std::stoul(option.substr(7)));
            } else if (option.rfind("--core=", 0) == 0) {
                core_label = option.substr(7);
                core = ParseCore(core_label);
            } else if (option.rfind("--speech-vocab=", 0) == 0) {
                speech_vocabulary_size = std::stoi(option.substr(15));
            } else if (option.rfind("--model-name=", 0) == 0) {
                model_name = option.substr(13);
            } else if (option.rfind("--max-context=", 0) == 0) {
                max_context = std::stoi(option.substr(14));
            } else if (option.rfind("--cpu-count=", 0) == 0) {
                enabled_cpus_num = std::stoi(option.substr(12));
            } else if (option.rfind("--cpu-mask=", 0) == 0) {
                enabled_cpus_mask = static_cast<uint32_t>(
                    std::stoul(option.substr(11), nullptr, 0));
            } else {
                throw std::runtime_error("unknown option: " + option);
            }
        }
        if (min_tokens < 0 || max_tokens <= min_tokens || max_context <= 0 ||
            enabled_cpus_num <= 0 || enabled_cpus_num > 8 ||
            enabled_cpus_mask == 0 || speech_vocabulary_size <= 0 ||
            model_name.empty()) {
            throw std::runtime_error("invalid token length limits");
        }

        auto speech_embedding = ReadBinary<float>(embedding_path);
        auto prefill = ReadBinary<float>(prefill_path);
        if (speech_embedding.empty() ||
            speech_embedding.size() % kHiddenSize != 0 ||
            prefill.empty() || prefill.size() % kHiddenSize != 0) {
            throw std::runtime_error("invalid embedding tensors");
        }
        const size_t embedding_vocabulary_size =
            speech_embedding.size() / kHiddenSize;
        if (embedding_vocabulary_size <
            static_cast<size_t>(speech_vocabulary_size)) {
            throw std::runtime_error("speech embedding vocabulary is too small");
        }
        const std::vector<int32_t> expected =
            expected_path.empty() ? std::vector<int32_t>{}
                                  : ReadBinary<int32_t>(expected_path);
        const std::vector<int32_t> teacher =
            teacher_path.empty() ? std::vector<int32_t>{}
                                 : ReadBinary<int32_t>(teacher_path);
        if (!teacher.empty()) max_tokens = teacher.size();

        const auto init_start = Clock::now();
        double rkllm_init_ms = 0.0;
        handle = InitializeRkllm(rkllm_model, max_context, enabled_cpus_num,
                                 enabled_cpus_mask, rkllm_init_ms);
        const auto head_init_start = Clock::now();
        SpeechHead speech_head(head_model, core, speech_vocabulary_size);
        if (speech_head.VocabularySize() != embedding_vocabulary_size) {
            throw std::runtime_error(
                "speech head and embedding vocabulary sizes differ");
        }
        const auto head_init_end = Clock::now();
        const auto head_warmup_start = Clock::now();
        speech_head.Warmup();
        const auto head_warmup_end = Clock::now();
        const auto init_end = Clock::now();

        std::vector<float> prefill_expected;
        if (!prefill_expected_path.empty()) {
            prefill_expected = ReadBinary<float>(prefill_expected_path);
        }
        const GenerationResult generated = GenerateCase(
            handle, speech_head, speech_embedding, prefill, expected, teacher,
            prefill_expected.empty() ? nullptr : &prefill_expected, min_tokens,
            max_tokens, speech_vocabulary_size, seed);
        WriteBinary(output_path, generated.generated);
        WriteBinary(output_path + ".sampled.i32.bin", generated.sampled);
        PrintGeneration(
            generated, teacher.empty() ? "sample" : "teacher", model_name,
            seed, core_label, speech_vocabulary_size,
            speech_head.VocabularySize(), Milliseconds(init_start, init_end),
            rkllm_init_ms, Milliseconds(head_init_start, head_init_end),
            Milliseconds(head_warmup_start, head_warmup_end), max_context,
            enabled_cpus_num, enabled_cpus_mask, output_path);

        rkllm_destroy(handle);
        return generated.generated.empty() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        if (handle != nullptr) rkllm_destroy(handle);
        return 1;
    }
}
