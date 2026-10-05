add_library(blocktensor_runtime STATIC
    src/dummy.cpp
    src/kv_cache.cpp
    src/transformer_engine.cpp
    src/tokenizer.cpp
)

target_include_directories(blocktensor_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(blocktensor_runtime PUBLIC
    blocktensor_quant
)
