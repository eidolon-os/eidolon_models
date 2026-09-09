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
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include "rknn_api.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int kSampleRate = 24000;
constexpr size_t kMelChannels = 80;
constexpr size_t kFftSize = 16;
constexpr size_t kFftBins = kFftSize / 2 + 1;
constexpr size_t kStftChannels = kFftBins * 2;
constexpr size_t kStftHop = 4;
constexpr size_t kF0Upsample = 480;
constexpr size_t kSourceCacheSamples = 3840;
constexpr size_t kHarmonics = 9;
constexpr size_t kNewFrames = 50;
constexpr size_t kContextFrames = 8;
constexpr float kSineAmplitude = 0.1f;
constexpr float kNoiseStd = 0.003f;
constexpr float kVoicedThreshold = 10.0f;
constexpr double kPi = 3.14159265358979323846;
constexpr unsigned kFftwEstimate = 64;

constexpr std::array<float, kHarmonics> kSourceLinearWeights = {
    -0.27458202838897705f, -0.2774406373500824f, 0.07214482128620148f,
    0.1259651780128479f,   0.02788151241838932f, 0.003079148707911372f,
    0.01020925771445036f,  -0.011415181681513786f,
    -0.013241725973784924f};
constexpr float kSourceLinearBias = 7.73382416809909e-05f;

double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void Check(int result, const std::string& operation) {
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

void WriteU16(std::ofstream& stream, uint16_t value) {
    const char bytes[2] = {static_cast<char>(value & 0xff),
                           static_cast<char>((value >> 8) & 0xff)};
    stream.write(bytes, sizeof(bytes));
}

void WriteU32(std::ofstream& stream, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xff),
        static_cast<char>((value >> 8) & 0xff),
        static_cast<char>((value >> 16) & 0xff),
        static_cast<char>((value >> 24) & 0xff)};
    stream.write(bytes, sizeof(bytes));
}

void WritePcm16Wav(const fs::path& path, const std::vector<float>& waveform) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to create " + path.string());
    const uint32_t data_bytes = static_cast<uint32_t>(waveform.size() * 2);
    stream.write("RIFF", 4);
    WriteU32(stream, 36 + data_bytes);
    stream.write("WAVEfmt ", 8);
    WriteU32(stream, 16);
    WriteU16(stream, 1);
    WriteU16(stream, 1);
    WriteU32(stream, kSampleRate);
    WriteU32(stream, kSampleRate * 2);
    WriteU16(stream, 2);
    WriteU16(stream, 16);
    stream.write("data", 4);
    WriteU32(stream, data_bytes);
    for (float sample : waveform) {
        sample = std::clamp(sample, -1.0f, 1.0f);
        const int16_t pcm = static_cast<int16_t>(std::lrint(
            sample < 0.0f ? sample * 32768.0f : sample * 32767.0f));
        WriteU16(stream, static_cast<uint16_t>(pcm));
    }
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

bool AllFinite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
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

struct CaseRecord {
    std::string case_id;
    size_t valid_tokens = 0;
    size_t generated_tokens = 0;
    size_t valid_frames = 0;
    uint32_t seed = 0;
    size_t target_samples = 0;
};

std::vector<std::string> Split(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, delimiter)) fields.push_back(field);
    return fields;
}

std::vector<CaseRecord> ReadManifest(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("failed to open " + path.string());
    std::vector<CaseRecord> rows;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.rfind("case_id\t", 0) == 0) continue;
        const auto fields = Split(line, '\t');
        if (fields.size() != 6) {
            throw std::runtime_error("invalid manifest row: " + line);
        }
        CaseRecord row;
        row.case_id = fields[0];
        row.valid_tokens = std::stoull(fields[1]);
        row.generated_tokens = std::stoull(fields[2]);
        row.valid_frames = std::stoull(fields[3]);
        row.seed = static_cast<uint32_t>(std::stoull(fields[4]));
        row.target_samples = std::stoull(fields[5]);
        if (row.target_samples != row.generated_tokens * 2 * kF0Upsample) {
            throw std::runtime_error("invalid target samples for " + row.case_id);
        }
        rows.push_back(row);
    }
    if (rows.empty()) throw std::runtime_error("empty board manifest");
    return rows;
}

class FftwApi {
   public:
    using Complex = float[2];
    using Plan = void*;
    using PlanDftR2C = Plan (*)(int, float*, Complex*, unsigned);
    using PlanDftC2R = Plan (*)(int, Complex*, float*, unsigned);
    using Execute = void (*)(const Plan);
    using DestroyPlan = void (*)(Plan);

    FftwApi() {
        library_ = dlopen("libfftw3f.so.3", RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            throw std::runtime_error("failed to load libfftw3f.so.3");
        }
        plan_dft_r2c = Load<PlanDftR2C>("fftwf_plan_dft_r2c_1d");
        plan_dft_c2r = Load<PlanDftC2R>("fftwf_plan_dft_c2r_1d");
        execute = Load<Execute>("fftwf_execute");
        destroy_plan = Load<DestroyPlan>("fftwf_destroy_plan");
    }

    ~FftwApi() {
        if (library_ != nullptr) dlclose(library_);
    }

    PlanDftR2C plan_dft_r2c = nullptr;
    PlanDftC2R plan_dft_c2r = nullptr;
    Execute execute = nullptr;
    DestroyPlan destroy_plan = nullptr;

   private:
    template <typename T>
    T Load(const char* name) {
        void* symbol = dlsym(library_, name);
        if (symbol == nullptr) {
            throw std::runtime_error(std::string("missing FFTW symbol ") + name);
        }
        return reinterpret_cast<T>(symbol);
    }

    void* library_ = nullptr;
};

class StftProcessor {
   public:
    StftProcessor() {
        for (size_t index = 0; index < kFftSize; ++index) {
            window_[index] = static_cast<float>(
                0.5 - 0.5 * std::cos(2.0 * kPi * index / kFftSize));
        }
        forward_plan_ = api_.plan_dft_r2c(
            static_cast<int>(kFftSize), forward_frame_.data(),
            forward_spectrum_.data(), kFftwEstimate);
        inverse_plan_ = api_.plan_dft_c2r(
            static_cast<int>(kFftSize), inverse_spectrum_.data(),
            inverse_frame_.data(), kFftwEstimate);
        if (forward_plan_ == nullptr || inverse_plan_ == nullptr) {
            throw std::runtime_error("failed to create FFTW plans");
        }
    }

    ~StftProcessor() {
        if (forward_plan_ != nullptr) api_.destroy_plan(forward_plan_);
        if (inverse_plan_ != nullptr) api_.destroy_plan(inverse_plan_);
    }

    void Forward(const std::vector<float>& source,
                 std::vector<float>& destination) {
        if (source.size() <= kFftSize / 2) {
            throw std::runtime_error("source is too short for STFT");
        }
        const size_t frames = source.size() / kStftHop + 1;
        if (destination.size() != kStftChannels * frames) {
            throw std::runtime_error("STFT destination size mismatch");
        }
        const int64_t samples = static_cast<int64_t>(source.size());
        const int64_t center = static_cast<int64_t>(kFftSize / 2);
        for (size_t time = 0; time < frames; ++time) {
            for (size_t index = 0; index < kFftSize; ++index) {
                int64_t source_index =
                    static_cast<int64_t>(time * kStftHop + index) - center;
                if (source_index < 0) {
                    source_index = -source_index;
                } else if (source_index >= samples) {
                    source_index = 2 * samples - 2 - source_index;
                }
                forward_frame_[index] =
                    source[static_cast<size_t>(source_index)] * window_[index];
            }
            api_.execute(forward_plan_);
            for (size_t bin = 0; bin < kFftBins; ++bin) {
                destination[bin * frames + time] = forward_spectrum_[bin][0];
                destination[(kFftBins + bin) * frames + time] =
                    forward_spectrum_[bin][1];
            }
        }
    }

    std::vector<float> Inverse(const std::vector<float>& decoder_raw,
                               size_t frames) {
        if (decoder_raw.size() != kStftChannels * frames || frames < 2) {
            throw std::runtime_error("invalid HiFT decoder output");
        }
        const size_t overlap_size = (frames - 1) * kStftHop + kFftSize;
        overlap_.assign(overlap_size, 0.0f);
        energy_.assign(overlap_size, 0.0f);
        for (size_t time = 0; time < frames; ++time) {
            for (size_t bin = 0; bin < kFftBins; ++bin) {
                const float magnitude =
                    std::min(std::exp(decoder_raw[bin * frames + time]), 100.0f);
                const float phase =
                    std::sin(decoder_raw[(kFftBins + bin) * frames + time]);
                inverse_spectrum_[bin][0] = magnitude * std::cos(phase);
                inverse_spectrum_[bin][1] = magnitude * std::sin(phase);
            }
            api_.execute(inverse_plan_);
            const size_t offset = time * kStftHop;
            for (size_t index = 0; index < kFftSize; ++index) {
                overlap_[offset + index] +=
                    inverse_frame_[index] * window_[index] / kFftSize;
                energy_[offset + index] += window_[index] * window_[index];
            }
        }
        const size_t output_size = (frames - 1) * kStftHop;
        std::vector<float> output(output_size);
        for (size_t index = 0; index < output_size; ++index) {
            const size_t source_index = index + kFftSize / 2;
            if (energy_[source_index] <= 1e-11f) {
                throw std::runtime_error("invalid iSTFT overlap energy");
            }
            output[index] = std::clamp(
                overlap_[source_index] / energy_[source_index], -0.99f, 0.99f);
        }
        return output;
    }

   private:
    FftwApi api_;
    std::array<float, kFftSize> window_{};
    std::array<float, kFftSize> forward_frame_{};
    std::array<FftwApi::Complex, kFftBins> forward_spectrum_{};
    std::array<FftwApi::Complex, kFftBins> inverse_spectrum_{};
    std::array<float, kFftSize> inverse_frame_{};
    FftwApi::Plan forward_plan_ = nullptr;
    FftwApi::Plan inverse_plan_ = nullptr;
    std::vector<float> overlap_;
    std::vector<float> energy_;
};

struct RknnTiming {
    double bridge_ms = 0.0;
    double sync_ms = 0.0;
    double npu_ms = 0.0;
    double readback_ms = 0.0;
};

class F0Bucket {
   public:
    F0Bucket(const fs::path& model, size_t frames, rknn_core_mask core)
        : frames_(frames) {
        Check(rknn_init(&context_, const_cast<char*>(model.c_str()), 0, 0,
                        nullptr),
              "rknn_init(F0)");
        Check(rknn_set_core_mask(context_, core), "rknn_set_core_mask(F0)");
        input_attr_.index = 0;
        output_attr_.index = 0;
        Check(rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr_,
                         sizeof(input_attr_)),
              "rknn_query(F0 input)");
        Check(rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attr_,
                         sizeof(output_attr_)),
              "rknn_query(F0 output)");
        if (input_attr_.type != RKNN_TENSOR_FLOAT16 ||
            output_attr_.type != RKNN_TENSOR_FLOAT16 ||
            input_attr_.n_elems != kMelChannels * frames_ ||
            output_attr_.n_elems != frames_ ||
            input_attr_.size_with_stride !=
                input_attr_.n_elems * sizeof(uint16_t)) {
            throw std::runtime_error("unexpected native F0 tensors");
        }
        input_ = rknn_create_mem(context_, input_attr_.size_with_stride);
        output_ = rknn_create_mem(context_, output_attr_.size_with_stride);
        if (input_ == nullptr || output_ == nullptr) {
            throw std::runtime_error("rknn_create_mem(F0) failed");
        }
        Check(rknn_set_io_mem(context_, input_, &input_attr_),
              "rknn_set_io_mem(F0 input)");
        Check(rknn_set_io_mem(context_, output_, &output_attr_),
              "rknn_set_io_mem(F0 output)");
    }

    ~F0Bucket() {
        if (input_ != nullptr) rknn_destroy_mem(context_, input_);
        if (output_ != nullptr) rknn_destroy_mem(context_, output_);
        if (context_ != 0) rknn_destroy(context_);
    }

    void Warmup() {
        std::memset(input_->virt_addr, 0, input_->size);
        Check(rknn_mem_sync(context_, input_, RKNN_MEMORY_SYNC_TO_DEVICE),
              "rknn_mem_sync(F0 warmup)");
        Check(rknn_run(context_, nullptr), "rknn_run(F0 warmup)");
    }

    RknnTiming Run(const std::vector<float>& mel, std::vector<float>& f0) {
        if (mel.size() != kMelChannels * frames_) {
            throw std::runtime_error("F0 mel size mismatch");
        }
        RknnTiming timing;
        const auto bridge_start = Clock::now();
        auto* input = static_cast<uint16_t*>(input_->virt_addr);
        for (size_t index = 0; index < mel.size(); ++index) {
            input[index] = FloatToHalf(mel[index]);
        }
        const auto bridge_end = Clock::now();
        Check(rknn_mem_sync(context_, input_, RKNN_MEMORY_SYNC_TO_DEVICE),
              "rknn_mem_sync(F0 input)");
        const auto sync_end = Clock::now();
        Check(rknn_run(context_, nullptr), "rknn_run(F0)");
        const auto run_end = Clock::now();
        Check(rknn_mem_sync(context_, output_, RKNN_MEMORY_SYNC_FROM_DEVICE),
              "rknn_mem_sync(F0 output)");
        const auto* output = static_cast<const uint16_t*>(output_->virt_addr);
        f0.resize(frames_);
        for (size_t index = 0; index < frames_; ++index) {
            f0[index] = HalfToFloat(output[index]);
        }
        const auto read_end = Clock::now();
        timing.bridge_ms = Milliseconds(bridge_start, bridge_end);
        timing.sync_ms = Milliseconds(bridge_end, sync_end);
        timing.npu_ms = Milliseconds(sync_end, run_end);
        timing.readback_ms = Milliseconds(run_end, read_end);
        return timing;
    }

   private:
    size_t frames_ = 0;
    rknn_context context_ = 0;
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr output_attr_{};
    rknn_tensor_mem* input_ = nullptr;
    rknn_tensor_mem* output_ = nullptr;
};

class DecoderBucket {
   public:
    DecoderBucket(const fs::path& model, size_t mel_frames,
                  rknn_core_mask core)
        : mel_frames_(mel_frames),
          stft_frames_(mel_frames_ * kF0Upsample / kStftHop + 1) {
        Check(rknn_init(&context_, const_cast<char*>(model.c_str()), 0, 0,
                        nullptr),
              "rknn_init(decoder)");
        Check(rknn_set_core_mask(context_, core),
              "rknn_set_core_mask(decoder)");
        for (uint32_t index = 0; index < input_attrs_.size(); ++index) {
            input_attrs_[index].index = index;
            Check(rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR,
                             &input_attrs_[index], sizeof(rknn_tensor_attr)),
                  "rknn_query(decoder input)");
        }
        output_attr_.index = 0;
        Check(rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attr_,
                         sizeof(output_attr_)),
              "rknn_query(decoder output)");
        if (input_attrs_[0].type != RKNN_TENSOR_FLOAT16 ||
            input_attrs_[1].type != RKNN_TENSOR_FLOAT16 ||
            input_attrs_[0].n_elems != kMelChannels * mel_frames_ ||
            input_attrs_[1].n_elems != kStftChannels * stft_frames_ ||
            input_attrs_[0].size_with_stride !=
                input_attrs_[0].n_elems * sizeof(uint16_t) ||
            input_attrs_[1].size_with_stride !=
                input_attrs_[1].n_elems * sizeof(uint16_t)) {
            throw std::runtime_error("unexpected native decoder inputs");
        }
        if (output_attr_.type != RKNN_TENSOR_FLOAT16 ||
            output_attr_.fmt != RKNN_TENSOR_NC1HWC2 ||
            output_attr_.n_dims != 5 || output_attr_.dims[4] != 8) {
            throw std::runtime_error("unexpected native decoder output");
        }
        for (size_t index = 0; index < inputs_.size(); ++index) {
            inputs_[index] = rknn_create_mem(
                context_, input_attrs_[index].size_with_stride);
            if (inputs_[index] == nullptr) {
                throw std::runtime_error("rknn_create_mem(decoder input) failed");
            }
            Check(rknn_set_io_mem(context_, inputs_[index], &input_attrs_[index]),
                  "rknn_set_io_mem(decoder input)");
        }
        output_ = rknn_create_mem(context_, output_attr_.size_with_stride);
        if (output_ == nullptr) {
            throw std::runtime_error("rknn_create_mem(decoder output) failed");
        }
        Check(rknn_set_io_mem(context_, output_, &output_attr_),
              "rknn_set_io_mem(decoder output)");
    }

    ~DecoderBucket() {
        for (auto* input : inputs_) {
            if (input != nullptr) rknn_destroy_mem(context_, input);
        }
        if (output_ != nullptr) rknn_destroy_mem(context_, output_);
        if (context_ != 0) rknn_destroy(context_);
    }

    void Warmup() {
        for (auto* input : inputs_) {
            std::memset(input->virt_addr, 0, input->size);
            Check(rknn_mem_sync(context_, input, RKNN_MEMORY_SYNC_TO_DEVICE),
                  "rknn_mem_sync(decoder warmup)");
        }
        Check(rknn_run(context_, nullptr), "rknn_run(decoder warmup)");
    }

    RknnTiming Run(const std::vector<float>& mel,
                   const std::vector<float>& source_stft,
                   std::vector<float>& decoder_raw) {
        if (mel.size() != kMelChannels * mel_frames_ ||
            source_stft.size() != kStftChannels * stft_frames_) {
            throw std::runtime_error("decoder input size mismatch");
        }
        RknnTiming timing;
        const auto bridge_start = Clock::now();
        const std::array<const std::vector<float>*, 2> values = {
            &mel, &source_stft};
        for (size_t tensor = 0; tensor < inputs_.size(); ++tensor) {
            auto* destination =
                static_cast<uint16_t*>(inputs_[tensor]->virt_addr);
            for (size_t index = 0; index < values[tensor]->size(); ++index) {
                destination[index] = FloatToHalf((*values[tensor])[index]);
            }
        }
        const auto bridge_end = Clock::now();
        for (auto* input : inputs_) {
            Check(rknn_mem_sync(context_, input, RKNN_MEMORY_SYNC_TO_DEVICE),
                  "rknn_mem_sync(decoder input)");
        }
        const auto sync_end = Clock::now();
        Check(rknn_run(context_, nullptr), "rknn_run(decoder)");
        const auto run_end = Clock::now();
        Check(rknn_mem_sync(context_, output_, RKNN_MEMORY_SYNC_FROM_DEVICE),
              "rknn_mem_sync(decoder output)");
        const size_t blocks = output_attr_.dims[1];
        const size_t height = output_attr_.dims[2];
        const size_t width = output_attr_.dims[3];
        const size_t lanes = output_attr_.dims[4];
        if (height != 1 || blocks * lanes < kStftChannels ||
            width < stft_frames_) {
            throw std::runtime_error("invalid native decoder layout");
        }
        const auto* native = static_cast<const uint16_t*>(output_->virt_addr);
        decoder_raw.resize(kStftChannels * stft_frames_);
        for (size_t channel = 0; channel < kStftChannels; ++channel) {
            const size_t block = channel / lanes;
            const size_t lane = channel % lanes;
            for (size_t frame = 0; frame < stft_frames_; ++frame) {
                const size_t native_index =
                    ((block * height) * width + frame) * lanes + lane;
                decoder_raw[channel * stft_frames_ + frame] =
                    HalfToFloat(native[native_index]);
            }
        }
        const auto read_end = Clock::now();
        timing.bridge_ms = Milliseconds(bridge_start, bridge_end);
        timing.sync_ms = Milliseconds(bridge_end, sync_end);
        timing.npu_ms = Milliseconds(sync_end, run_end);
        timing.readback_ms = Milliseconds(run_end, read_end);
        return timing;
    }

    size_t StftFrames() const { return stft_frames_; }

   private:
    size_t mel_frames_ = 0;
    size_t stft_frames_ = 0;
    rknn_context context_ = 0;
    std::array<rknn_tensor_attr, 2> input_attrs_{};
    rknn_tensor_attr output_attr_{};
    std::array<rknn_tensor_mem*, 2> inputs_{};
    rknn_tensor_mem* output_ = nullptr;
};

std::vector<float> GenerateNsfSource(const std::vector<float>& f0,
                                     std::mt19937& generator,
                                     const std::vector<float>& cache_source,
                                     int requested_threads) {
    const size_t samples = f0.size() * kF0Upsample;
    std::vector<float> source(samples, kSourceLinearBias);
    std::uniform_real_distribution<float> phase_distribution(
        static_cast<float>(-kPi), static_cast<float>(kPi));
    std::array<float, kHarmonics> phase{};
    for (float& value : phase) value = phase_distribution(generator);
    phase[0] = 0.0f;
    const size_t workers_count = std::min(
        f0.size(), static_cast<size_t>(std::max(requested_threads, 1)));
    std::vector<uint32_t> seeds(workers_count);
    for (uint32_t& value : seeds) value = generator();
    std::vector<double> prefix(f0.size() + 1, 0.0);
    for (size_t index = 0; index < f0.size(); ++index) {
        prefix[index + 1] = prefix[index] + f0[index];
    }
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < workers_count; ++worker) {
        workers.emplace_back([&, worker]() {
            const size_t begin = f0.size() * worker / workers_count;
            const size_t end = f0.size() * (worker + 1) / workers_count;
            std::mt19937 local(seeds[worker]);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (size_t harmonic = 0; harmonic < kHarmonics; ++harmonic) {
                const double multiplier = static_cast<double>(harmonic + 1);
                double angle = static_cast<double>(phase[harmonic]) +
                               2.0 * kPi * multiplier * kF0Upsample *
                                   prefix[begin] / kSampleRate;
                size_t sample = begin * kF0Upsample;
                for (size_t frame = begin; frame < end; ++frame) {
                    const float frequency = f0[frame];
                    const double delta =
                        2.0 * kPi * frequency * multiplier / kSampleRate;
                    const double sin_delta = std::sin(delta);
                    const double cos_delta = std::cos(delta);
                    double sin_angle = std::sin(angle);
                    double cos_angle = std::cos(angle);
                    const bool voiced = frequency > kVoicedThreshold;
                    const float noise_scale =
                        voiced ? kNoiseStd : kSineAmplitude / 3.0f;
                    for (size_t offset = 0; offset < kF0Upsample;
                         ++offset, ++sample) {
                        const double next_sine =
                            sin_angle * cos_delta + cos_angle * sin_delta;
                        const double next_cosine =
                            cos_angle * cos_delta - sin_angle * sin_delta;
                        sin_angle = next_sine;
                        cos_angle = next_cosine;
                        const float sine =
                            kSineAmplitude * static_cast<float>(sin_angle);
                        const float value =
                            (voiced ? sine : 0.0f) + noise_scale * normal(local);
                        source[sample] +=
                            kSourceLinearWeights[harmonic] * value;
                    }
                    angle = std::remainder(
                        angle + delta * static_cast<double>(kF0Upsample),
                        2.0 * kPi);
                }
            }
            for (size_t sample = begin * kF0Upsample;
                 sample < end * kF0Upsample; ++sample) {
                source[sample] = std::tanh(source[sample]);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    if (!cache_source.empty()) {
        if (cache_source.size() != kSourceCacheSamples ||
            source.size() < cache_source.size()) {
            throw std::runtime_error("invalid NSF source cache");
        }
        std::copy(cache_source.begin(), cache_source.end(), source.begin());
    }
    return source;
}

std::vector<float> Tail(const std::vector<float>& values, size_t count) {
    if (values.size() < count) {
        throw std::runtime_error("tensor shorter than cache");
    }
    return std::vector<float>(values.end() - static_cast<ptrdiff_t>(count),
                              values.end());
}

class StreamAssembler {
   public:
    StreamAssembler() {
        window_.resize(kSourceCacheSamples * 2);
        for (size_t index = 0; index < window_.size(); ++index) {
            window_[index] = static_cast<float>(
                0.54 - 0.46 * std::cos(2.0 * kPi * index /
                                      static_cast<double>(window_.size() - 1)));
        }
    }

    void Append(std::vector<float> speech, bool final,
                std::vector<float>& stream) {
        if (!cache_.empty()) {
            if (cache_.size() != kSourceCacheSamples ||
                speech.size() < kSourceCacheSamples) {
                throw std::runtime_error("invalid speech overlap cache");
            }
            for (size_t index = 0; index < kSourceCacheSamples; ++index) {
                speech[index] =
                    speech[index] * window_[index] +
                    cache_[index] * window_[kSourceCacheSamples + index];
            }
        }
        if (final) {
            stream.insert(stream.end(), speech.begin(), speech.end());
        } else {
            cache_ = Tail(speech, kSourceCacheSamples);
            stream.insert(stream.end(), speech.begin(),
                          speech.end() -
                              static_cast<ptrdiff_t>(kSourceCacheSamples));
        }
    }

   private:
    std::vector<float> window_;
    std::vector<float> cache_;
};

std::vector<float> MakeChunk(const std::vector<float>& mel,
                             size_t target_frames, size_t call,
                             size_t frames) {
    if (mel.size() != kMelChannels * target_frames) {
        throw std::runtime_error("invalid target mel tensor");
    }
    const size_t start = call == 0 ? 0 : call * kNewFrames - kContextFrames;
    std::vector<float> chunk(kMelChannels * frames, 0.0f);
    for (size_t channel = 0; channel < kMelChannels; ++channel) {
        for (size_t frame = 0; frame < frames; ++frame) {
            const size_t source_frame = start + frame;
            if (source_frame < target_frames) {
                chunk[channel * frames + frame] =
                    mel[channel * target_frames + source_frame];
            }
        }
    }
    return chunk;
}

struct Totals {
    double f0_bridge_ms = 0.0;
    double f0_sync_ms = 0.0;
    double f0_npu_ms = 0.0;
    double f0_readback_ms = 0.0;
    double nsf_ms = 0.0;
    double stft_ms = 0.0;
    double decoder_bridge_ms = 0.0;
    double decoder_sync_ms = 0.0;
    double decoder_npu_ms = 0.0;
    double decoder_readback_ms = 0.0;
    double istft_ms = 0.0;
    double stitch_ms = 0.0;
    double output_gain_ms = 0.0;

    double Sum() const {
        return f0_bridge_ms + f0_sync_ms + f0_npu_ms + f0_readback_ms +
               nsf_ms + stft_ms + decoder_bridge_ms + decoder_sync_ms +
               decoder_npu_ms + decoder_readback_ms + istft_ms + stitch_ms +
               output_gain_ms;
    }
};

struct ChunkTiming {
    size_t call = 0;
    size_t input_frames = 0;
    size_t playable_samples = 0;
    size_t cumulative_playable_samples = 0;
    double start_ms = 0.0;
    double ready_ms = 0.0;
    double wall_ms = 0.0;
    double ready_interval_ms = 0.0;
    double chunk_rtf = 0.0;
    double deadline_margin_ms = 0.0;
    double buffer_after_ms = 0.0;
    double f0_npu_ms = 0.0;
    double decoder_npu_ms = 0.0;
    double cpu_dsp_ms = 0.0;
    bool underrun = false;
};

struct AudioQc {
    double peak = 0.0;
    double rms = 0.0;
    double clipped_ratio = 0.0;
};

AudioQc CheckAudio(const std::vector<float>& waveform) {
    if (waveform.empty() || !AllFinite(waveform)) {
        throw std::runtime_error("invalid generated waveform");
    }
    double squared = 0.0;
    size_t clipped = 0;
    AudioQc qc;
    for (float sample : waveform) {
        qc.peak = std::max(qc.peak, std::abs(static_cast<double>(sample)));
        squared += sample * sample;
        if (std::abs(sample) >= 0.999f) ++clipped;
    }
    qc.rms = std::sqrt(squared / waveform.size());
    qc.clipped_ratio = static_cast<double>(clipped) / waveform.size();
    if (qc.rms <= 1e-6) {
        throw std::runtime_error("generated waveform failed QC");
    }
    return qc;
}

double Percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

void WriteCaseJson(const fs::path& path, const CaseRecord& row, int steps,
                   size_t calls, double wall_ms, const Totals& totals,
                   const AudioQc& raw_qc, double output_gain,
                   const AudioQc& qc,
                   const std::vector<ChunkTiming>& chunks) {
    const double audio_seconds =
        static_cast<double>(row.target_samples) / kSampleRate;
    const double first_chunk_ready_ms = chunks.front().ready_ms;
    const double first_chunk_seconds =
        static_cast<double>(chunks.front().playable_samples) / kSampleRate;
    std::vector<double> steady_intervals;
    size_t underruns = 0;
    double max_gap_ms = 0.0;
    for (size_t index = 1; index < chunks.size(); ++index) {
        steady_intervals.push_back(chunks[index].ready_interval_ms);
        if (chunks[index].underrun) {
            ++underruns;
            max_gap_ms = std::max(max_gap_ms,
                                  -chunks[index].deadline_margin_ms);
        }
    }
    const double steady_audio_seconds = audio_seconds - first_chunk_seconds;
    const double steady_wall_ms = wall_ms - first_chunk_ready_ms;
    const double steady_rtf = steady_audio_seconds > 0.0
                                  ? steady_wall_ms /
                                        (steady_audio_seconds * 1000.0)
                                  : 0.0;
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to create " + path.string());
    stream << std::setprecision(10)
           << "{\n  \"status\": \"PASS\",\n"
           << "  \"case_id\": \"" << row.case_id << "\",\n"
           << "  \"variant\": \"step" << steps << "\",\n"
           << "  \"steps\": " << steps << ",\n"
           << "  \"calls\": " << calls << ",\n"
           << "  \"target_samples\": " << row.target_samples << ",\n"
           << "  \"audio_seconds\": " << audio_seconds << ",\n"
           << "  \"hift_wall_ms\": " << wall_ms << ",\n"
           << "  \"hift_rtf\": " << wall_ms / (audio_seconds * 1000.0) << ",\n"
           << "  \"first_chunk_ready_ms\": " << first_chunk_ready_ms << ",\n"
           << "  \"first_chunk_seconds\": " << first_chunk_seconds << ",\n"
           << "  \"steady_wall_ms\": " << steady_wall_ms << ",\n"
           << "  \"steady_rtf\": " << steady_rtf << ",\n"
           << "  \"chunk_interval_p50_ms\": "
           << (steady_intervals.empty() ? 0.0
                                        : Percentile(steady_intervals, 0.5))
           << ",\n"
           << "  \"chunk_interval_p90_ms\": "
           << (steady_intervals.empty() ? 0.0
                                        : Percentile(steady_intervals, 0.9))
           << ",\n"
           << "  \"chunk_interval_max_ms\": "
           << (steady_intervals.empty()
                   ? 0.0
                   : *std::max_element(steady_intervals.begin(),
                                       steady_intervals.end()))
           << ",\n"
           << "  \"underrun_count\": " << underruns << ",\n"
           << "  \"max_output_gap_ms\": " << max_gap_ms << ",\n"
           << "  \"f0_npu_ms\": " << totals.f0_npu_ms << ",\n"
           << "  \"nsf_ms\": " << totals.nsf_ms << ",\n"
           << "  \"stft_ms\": " << totals.stft_ms << ",\n"
           << "  \"decoder_npu_ms\": " << totals.decoder_npu_ms << ",\n"
           << "  \"istft_ms\": " << totals.istft_ms << ",\n"
           << "  \"output_gain_ms\": " << totals.output_gain_ms << ",\n"
           << "  \"raw_peak_abs\": " << raw_qc.peak << ",\n"
           << "  \"raw_clipped_ratio\": " << raw_qc.clipped_ratio << ",\n"
           << "  \"output_gain\": " << output_gain << ",\n"
           << "  \"peak_abs\": " << qc.peak << ",\n"
           << "  \"rms\": " << qc.rms << ",\n"
           << "  \"clipped_ratio\": " << qc.clipped_ratio << "\n}\n";
    if (!stream) throw std::runtime_error("failed to write " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr
            << "usage: cosyvoice2_hift_rknn_benchmark MODEL_ROOT MANIFEST_TSV "
               "MEL_ROOT OUTPUT_ROOT STEPS [--threads=N] "
               "[--core=auto|0|1|2|01|012|all] [--max-cases=N] "
               "[--warmup=N] [--exact-tail]\n";
        return 2;
    }
    try {
        const fs::path model_root = argv[1];
        const fs::path manifest = argv[2];
        const fs::path mel_root = argv[3];
        const fs::path output_root = argv[4];
        const int steps = std::stoi(argv[5]);
        if (steps != 5 && steps != 2 && steps != 1) {
            throw std::runtime_error("steps must be 5, 2, or 1");
        }
        int threads = 4;
        rknn_core_mask core = RKNN_NPU_CORE_2;
        std::string core_label = "2";
        size_t max_cases = 0;
        int warmups = 1;
        bool exact_tail = false;
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option.rfind("--threads=", 0) == 0) {
                threads = std::stoi(option.substr(10));
            } else if (option.rfind("--core=", 0) == 0) {
                core_label = option.substr(7);
                core = ParseCore(core_label);
            } else if (option.rfind("--max-cases=", 0) == 0) {
                max_cases = std::stoull(option.substr(12));
            } else if (option.rfind("--warmup=", 0) == 0) {
                warmups = std::stoi(option.substr(9));
            } else if (option == "--exact-tail") {
                exact_tail = true;
            } else {
                throw std::runtime_error("unknown option: " + option);
            }
        }
        if (threads < 1 || warmups < 0) {
            throw std::runtime_error("invalid benchmark options");
        }
        auto rows = ReadManifest(manifest);
        if (max_cases != 0 && rows.size() > max_cases) rows.resize(max_cases);
        fs::create_directories(output_root / "cases");

        const auto init_start = Clock::now();
        std::map<size_t, std::unique_ptr<F0Bucket>> f0;
        std::map<size_t, std::unique_ptr<DecoderBucket>> decoder;
        std::set<size_t> bucket_frames = {
            kNewFrames, kContextFrames + kNewFrames};
        if (exact_tail) {
            for (const auto& row : rows) {
                const size_t target_frames = row.generated_tokens * 2;
                const size_t calls =
                    (target_frames + kNewFrames - 1) / kNewFrames;
                const size_t final_start = (calls - 1) * kNewFrames;
                const size_t final_fresh = target_frames - final_start;
                const size_t final_frames =
                    calls == 1 ? final_fresh : kContextFrames + final_fresh;
                bucket_frames.insert(final_frames);
            }
        }
        for (size_t frames : bucket_frames) {
            f0.emplace(frames, std::make_unique<F0Bucket>(
                                   model_root /
                                       ("hift_f0_fp16_seq" +
                                        std::to_string(frames) + ".rknn"),
                                   frames, core));
            decoder.emplace(frames, std::make_unique<DecoderBucket>(
                                        model_root /
                                            ("hift_decoder_fp16_mel" +
                                             std::to_string(frames) + ".rknn"),
                                        frames, core));
        }
        StftProcessor stft;
        const double init_ms = Milliseconds(init_start, Clock::now());
        const auto warmup_start = Clock::now();
        for (int iteration = 0; iteration < warmups; ++iteration) {
            for (auto& entry : f0) entry.second->Warmup();
            for (auto& entry : decoder) entry.second->Warmup();
        }
        const double warmup_ms = Milliseconds(warmup_start, Clock::now());

        const fs::path tsv_path =
            output_root / ("hift_step" + std::to_string(steps) + ".tsv");
        std::ofstream tsv(tsv_path, std::ios::trunc);
        if (!tsv) throw std::runtime_error("failed to create " + tsv_path.string());
        const fs::path chunk_tsv_path =
            output_root / ("hift_step" + std::to_string(steps) + "_chunks.tsv");
        std::ofstream chunk_tsv(chunk_tsv_path, std::ios::trunc);
        if (!chunk_tsv) {
            throw std::runtime_error("failed to create " +
                                     chunk_tsv_path.string());
        }
        tsv << "case_id\tsteps\tcalls\ttarget_samples\taudio_seconds"
               "\tf0_bridge_ms\tf0_sync_ms\tf0_npu_ms\tf0_readback_ms"
               "\tnsf_ms\tstft_ms\tdecoder_bridge_ms\tdecoder_sync_ms"
               "\tdecoder_npu_ms\tdecoder_readback_ms\tistft_ms\tstitch_ms"
               "\toutput_gain_ms\tcomponent_sum_ms\thift_wall_ms\thift_rtf"
               "\traw_peak_abs\traw_clipped_ratio\toutput_gain\tpeak_abs\trms"
               "\tclipped_ratio\tfinite\n";
        chunk_tsv
            << "case_id\tsteps\tcall\tinput_frames\tplayable_samples"
               "\tcumulative_playable_samples\tstart_ms\tready_ms\twall_ms"
               "\tready_interval_ms\tchunk_rtf\tdeadline_margin_ms"
               "\tbuffer_after_ms\tf0_npu_ms\tdecoder_npu_ms\tcpu_dsp_ms"
               "\tunderrun\n";
        std::vector<double> wall_times;
        std::vector<double> rtfs;
        std::vector<double> decoder_times;
        double audio_total = 0.0;
        std::cout << std::setprecision(10);

        for (const auto& row : rows) {
            const size_t target_frames = row.generated_tokens * 2;
            const fs::path case_root = mel_root / "cases" / row.case_id;
            const fs::path mel_path =
                case_root / ("board_step" + std::to_string(steps) +
                             "_target_mel.f32.bin");
            const auto mel = ReadBinary<float>(mel_path);
            if (mel.size() != kMelChannels * target_frames || !AllFinite(mel)) {
                throw std::runtime_error("invalid mel for " + row.case_id);
            }
            const size_t calls =
                (target_frames + kNewFrames - 1) / kNewFrames;
            std::mt19937 generator(row.seed);
            std::vector<float> source_cache;
            std::vector<float> waveform;
            StreamAssembler assembler;
            Totals totals;
            std::vector<ChunkTiming> chunks;
            chunks.reserve(calls);
            const auto case_start = Clock::now();
            for (size_t call = 0; call < calls; ++call) {
                const auto call_start = Clock::now();
                const size_t playable_before =
                    std::min(waveform.size(), row.target_samples);
                const size_t frames =
                    exact_tail && call + 1 == calls
                        ? (call == 0
                               ? target_frames
                               : kContextFrames +
                                     (target_frames - call * kNewFrames))
                        : (call == 0 ? kNewFrames
                                     : kContextFrames + kNewFrames);
                const auto chunk =
                    MakeChunk(mel, target_frames, call, frames);
                std::vector<float> pitch;
                const RknnTiming f0_timing = f0.at(frames)->Run(chunk, pitch);
                totals.f0_bridge_ms += f0_timing.bridge_ms;
                totals.f0_sync_ms += f0_timing.sync_ms;
                totals.f0_npu_ms += f0_timing.npu_ms;
                totals.f0_readback_ms += f0_timing.readback_ms;

                const auto nsf_start = Clock::now();
                auto source = GenerateNsfSource(
                    pitch, generator, source_cache, threads);
                const double nsf_ms = Milliseconds(nsf_start, Clock::now());
                totals.nsf_ms += nsf_ms;
                std::vector<float> source_stft(
                    kStftChannels * decoder.at(frames)->StftFrames());
                const auto stft_start = Clock::now();
                stft.Forward(source, source_stft);
                const double stft_ms = Milliseconds(stft_start, Clock::now());
                totals.stft_ms += stft_ms;

                std::vector<float> decoder_raw;
                const RknnTiming decoder_timing =
                    decoder.at(frames)->Run(chunk, source_stft, decoder_raw);
                totals.decoder_bridge_ms += decoder_timing.bridge_ms;
                totals.decoder_sync_ms += decoder_timing.sync_ms;
                totals.decoder_npu_ms += decoder_timing.npu_ms;
                totals.decoder_readback_ms += decoder_timing.readback_ms;
                if (!AllFinite(decoder_raw)) {
                    throw std::runtime_error("non-finite decoder output for " +
                                             row.case_id);
                }
                const auto istft_start = Clock::now();
                auto call_waveform = stft.Inverse(
                    decoder_raw, decoder.at(frames)->StftFrames());
                const double istft_ms =
                    Milliseconds(istft_start, Clock::now());
                totals.istft_ms += istft_ms;
                const auto stitch_start = Clock::now();
                assembler.Append(std::move(call_waveform), call + 1 == calls,
                                 waveform);
                const double stitch_ms =
                    Milliseconds(stitch_start, Clock::now());
                totals.stitch_ms += stitch_ms;
                source_cache = Tail(source, kSourceCacheSamples);

                const auto call_ready = Clock::now();
                const size_t playable_after =
                    std::min(waveform.size(), row.target_samples);
                ChunkTiming chunk_timing;
                chunk_timing.call = call;
                chunk_timing.input_frames = frames;
                chunk_timing.playable_samples =
                    playable_after - playable_before;
                chunk_timing.cumulative_playable_samples = playable_after;
                chunk_timing.start_ms = Milliseconds(case_start, call_start);
                chunk_timing.ready_ms = Milliseconds(case_start, call_ready);
                chunk_timing.wall_ms = Milliseconds(call_start, call_ready);
                chunk_timing.ready_interval_ms =
                    call == 0
                        ? chunk_timing.ready_ms
                        : chunk_timing.ready_ms - chunks.back().ready_ms;
                const double chunk_audio_ms =
                    1000.0 * chunk_timing.playable_samples / kSampleRate;
                chunk_timing.chunk_rtf =
                    chunk_audio_ms > 0.0
                        ? chunk_timing.wall_ms / chunk_audio_ms
                        : 0.0;
                if (call != 0) {
                    const double deadline_ms =
                        chunks.front().ready_ms +
                        1000.0 * playable_before / kSampleRate;
                    chunk_timing.deadline_margin_ms =
                        deadline_ms - chunk_timing.ready_ms;
                    chunk_timing.underrun =
                        chunk_timing.deadline_margin_ms < 0.0;
                }
                chunk_timing.buffer_after_ms =
                    chunk_timing.deadline_margin_ms + chunk_audio_ms;
                chunk_timing.f0_npu_ms = f0_timing.npu_ms;
                chunk_timing.decoder_npu_ms = decoder_timing.npu_ms;
                chunk_timing.cpu_dsp_ms =
                    nsf_ms + stft_ms + istft_ms + stitch_ms;
                chunks.push_back(chunk_timing);
            }
            if (waveform.size() < row.target_samples) {
                throw std::runtime_error("short waveform for " + row.case_id);
            }
            waveform.resize(row.target_samples);
            const AudioQc raw_qc = CheckAudio(waveform);
            const double output_gain =
                raw_qc.peak > 0.99 ? 0.99 / raw_qc.peak : 1.0;
            const auto gain_start = Clock::now();
            if (output_gain < 1.0) {
                for (float& sample : waveform) {
                    sample *= static_cast<float>(output_gain);
                }
            }
            totals.output_gain_ms += Milliseconds(gain_start, Clock::now());
            const AudioQc qc = CheckAudio(waveform);
            const double wall_ms = Milliseconds(case_start, Clock::now());
            const fs::path output_case = output_root / "cases" / row.case_id;
            fs::create_directories(output_case);
            WritePcm16Wav(output_case /
                              ("board_step" + std::to_string(steps) + ".wav"),
                          waveform);
            WriteCaseJson(output_case /
                              ("board_step" + std::to_string(steps) +
                               "_hift.json"),
                          row, steps, calls, wall_ms, totals, raw_qc,
                          output_gain, qc, chunks);
            const double audio_seconds =
                static_cast<double>(row.target_samples) / kSampleRate;
            const double rtf = wall_ms / (audio_seconds * 1000.0);
            tsv << std::setprecision(10) << row.case_id << '\t' << steps << '\t'
                << calls << '\t' << row.target_samples << '\t' << audio_seconds
                << '\t' << totals.f0_bridge_ms << '\t' << totals.f0_sync_ms
                << '\t' << totals.f0_npu_ms << '\t' << totals.f0_readback_ms
                << '\t' << totals.nsf_ms << '\t' << totals.stft_ms << '\t'
                << totals.decoder_bridge_ms << '\t' << totals.decoder_sync_ms
                << '\t' << totals.decoder_npu_ms << '\t'
                << totals.decoder_readback_ms << '\t' << totals.istft_ms
                << '\t' << totals.stitch_ms << '\t' << totals.output_gain_ms
                << '\t' << totals.Sum() << '\t' << wall_ms << '\t' << rtf
                << '\t' << raw_qc.peak << '\t' << raw_qc.clipped_ratio << '\t'
                << output_gain << '\t' << qc.peak << '\t' << qc.rms << '\t'
                << qc.clipped_ratio << "\t1\n";
            tsv.flush();
            for (const auto& chunk : chunks) {
                chunk_tsv << std::setprecision(10) << row.case_id << '\t'
                          << steps << '\t' << chunk.call << '\t'
                          << chunk.input_frames << '\t'
                          << chunk.playable_samples << '\t'
                          << chunk.cumulative_playable_samples << '\t'
                          << chunk.start_ms << '\t' << chunk.ready_ms << '\t'
                          << chunk.wall_ms << '\t'
                          << chunk.ready_interval_ms << '\t'
                          << chunk.chunk_rtf << '\t'
                          << chunk.deadline_margin_ms << '\t'
                          << chunk.buffer_after_ms << '\t'
                          << chunk.f0_npu_ms << '\t'
                          << chunk.decoder_npu_ms << '\t'
                          << chunk.cpu_dsp_ms << '\t'
                          << (chunk.underrun ? 1 : 0) << '\n';
            }
            chunk_tsv.flush();
            wall_times.push_back(wall_ms);
            rtfs.push_back(rtf);
            decoder_times.push_back(totals.decoder_npu_ms);
            audio_total += audio_seconds;
            std::cout << "case=" << row.case_id << " steps=" << steps
                      << " calls=" << calls << " hift_ms=" << wall_ms
                      << " hift_rtf=" << rtf << " rms=" << qc.rms << '\n';
        }
        const auto Mean = [](const std::vector<double>& values) {
            return std::accumulate(values.begin(), values.end(), 0.0) /
                   values.size();
        };
        std::cout << "status=PASS\n"
                  << "component=cosyvoice2_hift_all_rknn\n"
                  << "steps=" << steps << '\n'
                  << "cases=" << rows.size() << '\n'
                  << "audio_seconds_total=" << audio_total << '\n'
                  << "threads=" << threads << '\n'
                  << "npu_core=" << core_label << '\n'
                  << "exact_tail=" << (exact_tail ? 1 : 0) << '\n'
                  << "rknn_native_io_mem=1\n"
                  << "model_init_ms=" << init_ms << '\n'
                  << "warmup_ms=" << warmup_ms << '\n'
                  << "hift_wall_ms_mean=" << Mean(wall_times) << '\n'
                  << "hift_wall_ms_p50=" << Percentile(wall_times, 0.5) << '\n'
                  << "hift_wall_ms_p90=" << Percentile(wall_times, 0.9) << '\n'
                  << "hift_rtf_mean=" << Mean(rtfs) << '\n'
                  << "hift_rtf_total="
                  << std::accumulate(wall_times.begin(), wall_times.end(), 0.0) /
                         (audio_total * 1000.0)
                  << '\n'
                  << "decoder_npu_ms_mean=" << Mean(decoder_times) << '\n'
                  << "result_tsv=" << tsv_path.string() << '\n'
                  << "chunk_result_tsv=" << chunk_tsv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
