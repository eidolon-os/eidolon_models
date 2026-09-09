#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "cosyvoice2_text_frontend.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace cosyvoice2 {
namespace {

namespace fs = std::filesystem;

constexpr std::array<unsigned char, 8> kTokenizerMagic = {
    'C', 'V', '2', 'B', 'P', 'E', '1', 0};
constexpr const char* kPretokenPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

class MappedFile {
   public:
    explicit MappedFile(const fs::path& path) : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) Fail("open");
        struct stat info {};
        if (fstat(fd_, &info) != 0) Fail("fstat");
        if (info.st_size <= 0) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("empty mapped file: " + path_.string());
        }
        size_ = static_cast<std::size_t>(info.st_size);
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            Fail("mmap");
        }
    }

    ~MappedFile() {
        if (data_ != nullptr) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const unsigned char* Bytes() const {
        return static_cast<const unsigned char*>(data_);
    }
    const float* Floats() const {
        return static_cast<const float*>(data_);
    }
    std::size_t Size() const { return size_; }

   private:
    [[noreturn]] void Fail(const char* operation) {
        const int error = errno;
        if (data_ != nullptr) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        throw std::runtime_error(std::string(operation) + " failed for " +
                                 path_.string() + ": " +
                                 std::strerror(error));
    }

    fs::path path_;
    int fd_ = -1;
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

class BinaryReader {
   public:
    explicit BinaryReader(const MappedFile& file)
        : data_(file.Bytes()), size_(file.Size()) {}

    uint32_t U32() {
        Require(4);
        const uint32_t value = static_cast<uint32_t>(data_[offset_]) |
                               (static_cast<uint32_t>(data_[offset_ + 1]) << 8) |
                               (static_cast<uint32_t>(data_[offset_ + 2]) << 16) |
                               (static_cast<uint32_t>(data_[offset_ + 3]) << 24);
        offset_ += 4;
        return value;
    }

    std::string String() {
        const std::size_t length = U32();
        Require(length);
        std::string value(reinterpret_cast<const char*>(data_ + offset_), length);
        offset_ += length;
        return value;
    }

    void Magic() {
        Require(kTokenizerMagic.size());
        if (!std::equal(kTokenizerMagic.begin(), kTokenizerMagic.end(),
                        data_ + offset_)) {
            throw std::runtime_error("invalid CosyVoice2 tokenizer magic");
        }
        offset_ += kTokenizerMagic.size();
    }

    void RequireEnd() const {
        if (offset_ != size_) {
            throw std::runtime_error("trailing data in CosyVoice2 tokenizer");
        }
    }

   private:
    void Require(std::size_t count) const {
        if (count > size_ - offset_) {
            throw std::runtime_error("truncated CosyVoice2 tokenizer");
        }
    }

    const unsigned char* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

std::string EncodeCodepoint(uint32_t value) {
    std::string result;
    if (value <= 0x7f) {
        result.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        result.push_back(static_cast<char>(0xc0 | (value >> 6)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        result.push_back(static_cast<char>(0xe0 | (value >> 12)));
        result.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        result.push_back(static_cast<char>(0xf0 | (value >> 18)));
        result.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    return result;
}

std::array<std::string, 256> MakeByteEncoder() {
    std::array<bool, 256> direct{};
    for (int value = 33; value <= 126; ++value) direct[value] = true;
    for (int value = 161; value <= 172; ++value) direct[value] = true;
    for (int value = 174; value <= 255; ++value) direct[value] = true;

    std::array<std::string, 256> encoder;
    uint32_t extra = 0;
    for (uint32_t value = 0; value < 256; ++value) {
        encoder[value] = EncodeCodepoint(
            direct[value] ? value : static_cast<uint32_t>(256 + extra++));
    }
    return encoder;
}

std::vector<std::string> SplitUtf8Symbols(const std::string& text) {
    std::vector<std::string> result;
    for (std::size_t offset = 0; offset < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[offset]);
        std::size_t length = 0;
        if ((lead & 0x80) == 0) {
            length = 1;
        } else if ((lead & 0xe0) == 0xc0) {
            length = 2;
        } else if ((lead & 0xf0) == 0xe0) {
            length = 3;
        } else if ((lead & 0xf8) == 0xf0) {
            length = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 byte encoder output");
        }
        if (length > text.size() - offset) {
            throw std::runtime_error("truncated UTF-8 byte encoder output");
        }
        result.emplace_back(text.substr(offset, length));
        offset += length;
    }
    return result;
}

struct Pair {
    std::string left;
    std::string right;

    bool operator==(const Pair& other) const {
        return left == other.left && right == other.right;
    }
};

struct PairHash {
    std::size_t operator()(const Pair& value) const {
        const std::size_t first = std::hash<std::string>{}(value.left);
        const std::size_t second = std::hash<std::string>{}(value.right);
        return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6) +
                        (first >> 2));
    }
};

std::vector<int32_t> ReadI32(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("failed to open " + path.string());
    const std::streamoff bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("invalid int32 file: " + path.string());
    }
    std::vector<int32_t> result(static_cast<std::size_t>(bytes) /
                                sizeof(int32_t));
    stream.seekg(0);
    if (!result.empty()) {
        stream.read(reinterpret_cast<char*>(result.data()), bytes);
    }
    if (!stream) throw std::runtime_error("failed to read " + path.string());
    return result;
}

}  // namespace

class TextFrontend::Impl {
   public:
    Impl(const fs::path& model_root, const fs::path& voice_profile_root,
         const fs::path& speech_embedding_path)
        : tokenizer_file_(model_root / "tokenizer.bin"),
          text_embedding_(model_root / "text_embedding.f32.bin"),
          control_embedding_(model_root / "control_embedding.f32.bin"),
          speech_embedding_(speech_embedding_path),
          prompt_text_ids_(
              ReadI32(voice_profile_root / "prompt_text_token.i32.bin")),
          prompt_speech_ids_(ReadI32(
              voice_profile_root / "llm_prompt_speech_token.i32.bin")),
          byte_encoder_(MakeByteEncoder()) {
        LoadTokenizer();
        ValidateEmbedding(text_embedding_, "text embedding", &text_rows_);
        ValidateEmbedding(control_embedding_, "control embedding",
                          &control_rows_);
        ValidateEmbedding(speech_embedding_, "speech embedding", &speech_rows_);
        if (control_rows_ != 2) {
            throw std::runtime_error("control embedding must have exactly 2 rows");
        }
        if (prompt_text_ids_.empty() || prompt_speech_ids_.empty()) {
            throw std::runtime_error("voice profile token lists cannot be empty");
        }
        ValidateIds(prompt_text_ids_, text_rows_, "voice prompt text");
        ValidateIds(prompt_speech_ids_, speech_rows_, "voice prompt speech");

        int error_code = 0;
        PCRE2_SIZE error_offset = 0;
        regex_ = pcre2_compile(
            reinterpret_cast<PCRE2_SPTR>(kPretokenPattern), PCRE2_ZERO_TERMINATED,
            PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, nullptr);
        if (regex_ == nullptr) {
            std::array<PCRE2_UCHAR, 256> message{};
            pcre2_get_error_message(error_code, message.data(), message.size());
            throw std::runtime_error(
                "failed to compile Qwen2 pre-tokenizer at byte " +
                std::to_string(error_offset) + ": " +
                reinterpret_cast<const char*>(message.data()));
        }
    }

    ~Impl() {
        if (regex_ != nullptr) pcre2_code_free(regex_);
    }

    std::vector<int32_t> Tokenize(const std::string& text) const {
        if (text.empty()) throw std::runtime_error("target text is empty");
        std::vector<int32_t> result;
        std::size_t cursor = 0;
        while (cursor < text.size()) {
            std::size_t special_at = std::string::npos;
            const SpecialToken* selected = nullptr;
            for (const auto& special : specials_) {
                const std::size_t found = text.find(special.text, cursor);
                if (found == std::string::npos) continue;
                if (selected == nullptr || found < special_at ||
                    (found == special_at &&
                     special.text.size() > selected->text.size())) {
                    special_at = found;
                    selected = &special;
                }
            }
            if (selected == nullptr) {
                TokenizeOrdinary(std::string_view(text).substr(cursor), result);
                break;
            }
            if (special_at > cursor) {
                TokenizeOrdinary(std::string_view(text).substr(
                                     cursor, special_at - cursor),
                                 result);
            }
            result.push_back(selected->id);
            cursor = special_at + selected->text.size();
        }
        if (result.empty()) {
            throw std::runtime_error("target text produced no tokens");
        }
        return result;
    }

    TextFrontendResult BuildPrefill(const std::string& target_text) const {
        TextFrontendResult result;
        result.target_token_ids = Tokenize(target_text);
        ValidateIds(result.target_token_ids, text_rows_, "target text");
        const std::size_t rows = PrefillTokenCount(result.target_token_ids.size());
        if (rows > std::numeric_limits<std::size_t>::max() /
                       kTextFrontendHiddenSize) {
            throw std::runtime_error("prefill tensor is too large");
        }
        result.prefill.resize(rows * kTextFrontendHiddenSize);
        std::size_t output_row = 0;
        AppendRow(control_embedding_, 0, result.prefill, output_row++);
        AppendRows(text_embedding_, prompt_text_ids_, result.prefill, output_row);
        AppendRows(text_embedding_, result.target_token_ids, result.prefill,
                   output_row);
        AppendRow(control_embedding_, 1, result.prefill, output_row++);
        AppendRows(speech_embedding_, prompt_speech_ids_, result.prefill,
                   output_row);
        if (output_row != rows) {
            throw std::runtime_error("internal prefill row-count mismatch");
        }
        return result;
    }

    std::size_t PrefillTokenCount(std::size_t target_count) const {
        return 2 + prompt_text_ids_.size() + target_count +
               prompt_speech_ids_.size();
    }

    std::size_t PromptTextTokenCount() const { return prompt_text_ids_.size(); }
    std::size_t PromptSpeechTokenCount() const {
        return prompt_speech_ids_.size();
    }

   private:
    struct SpecialToken {
        int32_t id;
        std::string text;
    };

    static void ValidateEmbedding(const MappedFile& file, const char* label,
                                  std::size_t* rows) {
        constexpr std::size_t row_bytes =
            kTextFrontendHiddenSize * sizeof(float);
        if (file.Size() % row_bytes != 0) {
            throw std::runtime_error(std::string(label) +
                                     " has an invalid byte size");
        }
        *rows = file.Size() / row_bytes;
    }

    static void ValidateIds(const std::vector<int32_t>& ids,
                            std::size_t row_count, const char* label) {
        for (int32_t id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= row_count) {
                throw std::runtime_error(std::string(label) +
                                         " token is outside embedding table");
            }
        }
    }

    void LoadTokenizer() {
        BinaryReader reader(tokenizer_file_);
        reader.Magic();
        const uint32_t vocab_count = reader.U32();
        vocabulary_.reserve(vocab_count * 2);
        for (uint32_t index = 0; index < vocab_count; ++index) {
            const uint32_t id = reader.U32();
            const std::string token = reader.String();
            if (id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                !vocabulary_.emplace(token, static_cast<int32_t>(id)).second) {
                throw std::runtime_error("invalid or duplicate tokenizer token");
            }
        }
        const uint32_t merge_count = reader.U32();
        merge_ranks_.reserve(merge_count * 2);
        for (uint32_t rank = 0; rank < merge_count; ++rank) {
            Pair pair{reader.String(), reader.String()};
            if (!merge_ranks_.emplace(std::move(pair), rank).second) {
                throw std::runtime_error("duplicate tokenizer BPE merge");
            }
        }
        const uint32_t special_count = reader.U32();
        specials_.reserve(special_count);
        for (uint32_t index = 0; index < special_count; ++index) {
            const uint32_t id = reader.U32();
            std::string token = reader.String();
            if (token.empty() ||
                id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error("invalid tokenizer special token");
            }
            specials_.push_back(
                {static_cast<int32_t>(id), std::move(token)});
        }
        reader.RequireEnd();
    }

    void TokenizeOrdinary(std::string_view text,
                          std::vector<int32_t>& output) const {
        if (text.empty()) return;
        pcre2_match_data* match = pcre2_match_data_create_from_pattern(
            regex_, nullptr);
        if (match == nullptr) {
            throw std::runtime_error("failed to allocate PCRE2 match data");
        }
        struct MatchGuard {
            pcre2_match_data* value;
            ~MatchGuard() { pcre2_match_data_free(value); }
        } guard{match};

        std::size_t offset = 0;
        while (offset < text.size()) {
            const int status = pcre2_match(
                regex_, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
                offset, 0, match, nullptr);
            if (status < 0) {
                throw std::runtime_error(
                    status == PCRE2_ERROR_NOMATCH
                        ? "Qwen2 pre-tokenizer left unmatched input"
                        : "Qwen2 pre-tokenizer rejected invalid UTF-8 input: " +
                              std::to_string(status));
            }
            PCRE2_SIZE* range = pcre2_get_ovector_pointer(match);
            if (range[0] != offset || range[1] <= range[0]) {
                throw std::runtime_error(
                    "Qwen2 pre-tokenizer produced a non-contiguous match");
            }
            ApplyBpe(text.substr(range[0], range[1] - range[0]), output);
            offset = range[1];
        }
    }

    void ApplyBpe(std::string_view piece, std::vector<int32_t>& output) const {
        std::string encoded;
        encoded.reserve(piece.size() * 2);
        for (unsigned char byte : piece) encoded += byte_encoder_[byte];
        std::vector<std::string> symbols = SplitUtf8Symbols(encoded);
        while (symbols.size() > 1) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            Pair best;
            bool found = false;
            for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
                const Pair candidate{symbols[index], symbols[index + 1]};
                const auto rank = merge_ranks_.find(candidate);
                if (rank != merge_ranks_.end() && rank->second < best_rank) {
                    best_rank = rank->second;
                    best = candidate;
                    found = true;
                }
            }
            if (!found) break;

            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (std::size_t index = 0; index < symbols.size();) {
                if (index + 1 < symbols.size() && symbols[index] == best.left &&
                    symbols[index + 1] == best.right) {
                    merged.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(std::move(symbols[index]));
                    ++index;
                }
            }
            symbols = std::move(merged);
        }

        for (const std::string& symbol : symbols) {
            const auto token = vocabulary_.find(symbol);
            if (token == vocabulary_.end()) {
                throw std::runtime_error("BPE symbol is absent from vocabulary");
            }
            output.push_back(token->second);
        }
    }

    static void AppendRow(const MappedFile& source, std::size_t source_row,
                          std::vector<float>& destination,
                          std::size_t destination_row) {
        const float* begin =
            source.Floats() + source_row * kTextFrontendHiddenSize;
        std::memcpy(destination.data() +
                        destination_row * kTextFrontendHiddenSize,
                    begin, kTextFrontendHiddenSize * sizeof(float));
    }

    static void AppendRows(const MappedFile& source,
                           const std::vector<int32_t>& source_rows,
                           std::vector<float>& destination,
                           std::size_t& destination_row) {
        for (int32_t row : source_rows) {
            AppendRow(source, static_cast<std::size_t>(row), destination,
                      destination_row++);
        }
    }

    MappedFile tokenizer_file_;
    MappedFile text_embedding_;
    MappedFile control_embedding_;
    MappedFile speech_embedding_;
    std::vector<int32_t> prompt_text_ids_;
    std::vector<int32_t> prompt_speech_ids_;
    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<std::string, int32_t> vocabulary_;
    std::unordered_map<Pair, uint32_t, PairHash> merge_ranks_;
    std::vector<SpecialToken> specials_;
    pcre2_code* regex_ = nullptr;
    std::size_t text_rows_ = 0;
    std::size_t control_rows_ = 0;
    std::size_t speech_rows_ = 0;
};

TextFrontend::TextFrontend(const fs::path& model_root,
                           const fs::path& voice_profile_root,
                           const fs::path& speech_embedding_path)
    : impl_(std::make_unique<Impl>(model_root, voice_profile_root,
                                   speech_embedding_path)) {}

TextFrontend::~TextFrontend() = default;
TextFrontend::TextFrontend(TextFrontend&&) noexcept = default;
TextFrontend& TextFrontend::operator=(TextFrontend&&) noexcept = default;

std::vector<int32_t> TextFrontend::Tokenize(const std::string& text) const {
    return impl_->Tokenize(text);
}

TextFrontendResult TextFrontend::BuildPrefill(
    const std::string& target_text) const {
    return impl_->BuildPrefill(target_text);
}

std::size_t TextFrontend::PromptTextTokenCount() const {
    return impl_->PromptTextTokenCount();
}

std::size_t TextFrontend::PromptSpeechTokenCount() const {
    return impl_->PromptSpeechTokenCount();
}

std::size_t TextFrontend::PrefillTokenCount(
    std::size_t target_token_count) const {
    return impl_->PrefillTokenCount(target_token_count);
}

}  // namespace cosyvoice2
