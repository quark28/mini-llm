cmake_minimum_required(VERSION 3.20)
project(minillm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config/libs/onnxruntime-win-x64-1.31.1")
set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config/models")

include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(json)

add_executable(minillm src/main.cpp src/console.cpp src/model.cpp src/sampler.cpp src/tokenizer.cpp)
target_include_directories(minillm PRIVATE ${ORT_DIR}/include)
target_link_libraries(minillm PRIVATE ${ORT_DIR}/lib/onnxruntime.lib nlohmann_json::nlohmann_json)
target_compile_definitions(minillm PRIVATE MODELS_DIR="${MODELS_DIR}")

add_custom_command(TARGET minillm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DIR}/lib/onnxruntime.dll $<TARGET_FILE_DIR:minillm>)