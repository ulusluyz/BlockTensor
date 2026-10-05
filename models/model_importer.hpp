#pragma once

#include "blocktensor/transformer_weights.hpp"
#include "memory/memory_manager.hpp"
#include <string>

namespace blocktensor::importer {

void export_btmodel(const std::string& filepath, const TransformerWeights& weights, MemoryManager& mem);
TransformerWeights import_btmodel(const std::string& filepath, std::shared_ptr<IComputeBackend> backend, MemoryManager& mem);

} // namespace blocktensor::importer
