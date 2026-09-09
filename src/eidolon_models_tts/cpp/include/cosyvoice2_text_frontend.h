#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cosyvoice2 {

constexpr std::size_t kTextFrontendHiddenSize = 896;

struct TextFrontendResult {
    std::vector<int32_t> target_token_ids;
    std::vector<float> prefill;
};

class TextFrontend {
   public:
    TextFrontend(const std::filesystem::path& model_root,
                 const std::filesystem::path& voice_profile_root,
                 const std::filesystem::path& speech_embedding_path);
    ~TextFrontend();

    TextFrontend(TextFrontend&&) noexcept;
    TextFrontend& operator=(TextFrontend&&) noexcept;
    TextFrontend(const TextFrontend&) = delete;
    TextFrontend& operator=(const TextFrontend&) = delete;

    std::vector<int32_t> Tokenize(const std::string& text) const;
    TextFrontendResult BuildPrefill(const std::string& target_text) const;

    std::size_t PromptTextTokenCount() const;
    std::size_t PromptSpeechTokenCount() const;
    std::size_t PrefillTokenCount(std::size_t target_token_count) const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cosyvoice2
