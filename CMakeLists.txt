cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER "/cluster/apps/cuda/12.3/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/cluster/apps/cuda/12.3")
set(CUDA_TOOLKIT_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_TOOLKIT_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Add CUDA library path
link_directories(${CUDA_TOOLKIT_LIBRARY_DIRS})

option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Set CPM cache directory to avoid downloading dependencies every time
  set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/.cpm_cache)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# ---------------------------------
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# Find CUDA package
find_package(CUDA REQUIRED)

aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/op/ DIR_OP)
aux_source_directory(kuiper/source/model/ DIR_MODEL)
aux_source_directory(kuiper/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(kuiper/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(kuiper/source/op/kernels/ DIR_KERNEL)
aux_source_directory(kuiper/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} armadillo)
target_link_directories(llama PUBLIC ${CUDA_TOOLKIT_LIBRARY_DIRS})

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CUDA_INCLUDE_DIRS})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)
add_subdirectory(demo)
