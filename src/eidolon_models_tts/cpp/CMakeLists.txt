# The TTS engine, built on the Host that runs it.
#
# Not cross-compiled and not shipped as a binary: it links librknnrt and
# librkllmrt, which are the board's own NPU runtime, and a release is exact Git
# commits. So the source rides the release and `scripts/eidolon-tts-build`
# builds it once on the Host, next to the runtime it links against.
#
# The two NPU runtime libraries are found rather than vendored — see
# EIDOLON_TTS_RUNTIME_ROOT below.
cmake_minimum_required(VERSION 3.16)
project(eidolon_models_tts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Where librknnrt.so / librkllmrt.so and their headers are on this Host. The
# launcher passes it; the default is where the board's vendor packages put them.
set(EIDOLON_TTS_RUNTIME_ROOT "/usr/lib" CACHE PATH
    "Directory holding librknnrt.so and librkllmrt.so")
set(EIDOLON_TTS_RUNTIME_INCLUDE "" CACHE PATH
    "Directory holding rknn_api.h and rkllm.h")

find_library(RKNN_RUNTIME NAMES rknnrt HINTS "${EIDOLON_TTS_RUNTIME_ROOT}" REQUIRED)
find_library(RKLLM_RUNTIME NAMES rkllmrt HINTS "${EIDOLON_TTS_RUNTIME_ROOT}" REQUIRED)
# The text frontend needs PCRE2: CosyVoice2's normalisation is regex-driven and
# std::regex does not do UTF-8 properties.
find_path(PCRE2_INCLUDE_DIR NAMES pcre2.h REQUIRED)
find_library(PCRE2_8_LIBRARY NAMES pcre2-8 REQUIRED)

add_library(cosyvoice2_text_frontend STATIC src/cosyvoice2_text_frontend.cpp)
target_include_directories(cosyvoice2_text_frontend PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${PCRE2_INCLUDE_DIR}")
target_compile_definitions(cosyvoice2_text_frontend PUBLIC PCRE2_CODE_UNIT_WIDTH=8)
target_link_libraries(cosyvoice2_text_frontend PUBLIC "${PCRE2_8_LIBRARY}")

add_executable(cosyvoice2_streaming_pipeline src/cosyvoice2_streaming_pipeline.cpp)
target_include_directories(cosyvoice2_streaming_pipeline PRIVATE
    "${EIDOLON_TTS_RUNTIME_INCLUDE}")
target_link_libraries(cosyvoice2_streaming_pipeline PRIVATE
    "${RKLLM_RUNTIME}" "${RKNN_RUNTIME}" pthread dl cosyvoice2_text_frontend)
