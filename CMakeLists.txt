cmake_minimum_required(VERSION 3.20)
project(BlockTensor VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing BEFORE subdirectories
enable_testing()

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find glslangValidator
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator glslc DOC "glslangValidator executable")
if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator or glslc is required to compile GLSL compute shaders to SPIR-V")
endif()

message(STATUS "Found Vulkan: ${Vulkan_LIBRARIES}")
message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR_EXECUTABLE}")

# Include directories at root level
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${Vulkan_INCLUDE_DIRS}
)

# Shader compilation helper
function(target_compile_shaders TARGET_NAME SHADER_SOURCES)
    set(SPIRV_FILES "")
    foreach(SHADER_SRC ${SHADER_SOURCES})
        get_filename_component(FILE_NAME ${SHADER_SRC} NAME)
        set(SPIRV_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

        add_custom_command(
            OUTPUT ${SPIRV_OUT}
            COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V ${SHADER_SRC} -o ${SPIRV_OUT}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling GLSL shader ${FILE_NAME} to SPIR-V"
        )
        list(APPEND SPIRV_FILES ${SPIRV_OUT})
    endforeach()
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${SPIRV_FILES})
endfunction()

# Compile GLSL shaders
file(GLOB GLSL_SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/glsl/*.comp")
if(GLSL_SHADER_SOURCES)
    target_compile_shaders(compile_shaders "${GLSL_SHADER_SOURCES}")
endif()

# Include subdirectories
add_subdirectory(runtime)
add_subdirectory(backends)
add_subdirectory(models)
add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(benchmarks)
