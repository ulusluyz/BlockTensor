add_library(blocktensor_quant STATIC
    ../quantization/quant_types.cpp
)

target_include_directories(blocktensor_quant PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(blocktensor_backends STATIC
    cpu/cpu_backend.cpp
    vulkan/vulkan_backend.cpp
    ../memory/memory_manager.cpp
)

target_include_directories(blocktensor_backends PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(blocktensor_backends PUBLIC
    ${Vulkan_LIBRARIES}
    blocktensor_runtime
    blocktensor_quant
)
