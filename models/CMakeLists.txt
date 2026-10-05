add_library(blocktensor_models STATIC
    model_importer.cpp
)

target_include_directories(blocktensor_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(blocktensor_models PUBLIC
    blocktensor_runtime
    blocktensor_backends
)
