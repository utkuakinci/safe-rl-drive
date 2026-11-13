cmake_minimum_required(VERSION 3.18)
project(rl_policy_node LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

option(WITH_TENSORRT "Build the TensorRT backend (needs CUDA + TensorRT 10)" OFF)
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime C/C++ release directory")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session REQUIRED)
find_library(ORT_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)

# --- pure C++ shield (no ROS dependency, unit-tested) ---
add_library(safety_shield STATIC src/safety_shield.cpp)
target_include_directories(safety_shield PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
set_target_properties(safety_shield PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- inference backends ---
add_library(inference_backends SHARED src/onnx_backend.cpp)
target_include_directories(inference_backends PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${ORT_INCLUDE_DIR})
target_link_libraries(inference_backends ${ORT_LIBRARY})
if(WITH_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  find_library(NVINFER nvinfer REQUIRED)
  target_sources(inference_backends PRIVATE src/tensorrt_backend.cpp)
  target_compile_definitions(inference_backends PUBLIC WITH_TENSORRT)
  target_link_libraries(inference_backends ${NVINFER} CUDA::cudart)
endif()

# --- ROS2 component + standalone executable ---
add_library(policy_component SHARED src/policy_node.cpp)
target_link_libraries(policy_component inference_backends safety_shield)
ament_target_dependencies(policy_component rclcpp rclcpp_components std_msgs)
rclcpp_components_register_node(policy_component
  PLUGIN "rl_policy_node::PolicyNode"
  EXECUTABLE policy_node)

# --- latency micro-benchmark without ROS transport ---
add_executable(bench_inference src/bench_inference.cpp)
target_link_libraries(bench_inference inference_backends safety_shield)

install(TARGETS safety_shield inference_backends policy_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS bench_inference RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_safety_shield test/test_safety_shield.cpp)
  target_link_libraries(test_safety_shield safety_shield)
  target_compile_definitions(test_safety_shield PRIVATE
    SHIELD_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/test/shield_vectors.csv")
endif()

ament_package()
