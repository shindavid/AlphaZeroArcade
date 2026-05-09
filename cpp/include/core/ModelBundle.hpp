#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace core {

/*
 * ModelBundle encapsulates an ONNX model received from the loop-controller.
 *
 * The ONNX bytes are intended for TensorRT consumption. Some paradigms (e.g. BetaZero)
 * additionally embed auxiliary weights (e.g. NNUE backup-network weights) in the same ONNX
 * file as orphan initializers — TensorProtos in graph.initializer that are not consumed by
 * any node in the graph. TensorRT silently ignores these orphan initializers, but they are
 * accessible via the ONNX protobuf API.
 *
 * Convention: orphan initializers whose names start with "nnue/" are extracted into
 * nnue_weights, keyed by name with the "nnue/" prefix stripped. For paradigms that do not
 * use NNUE (e.g. AlphaZero), nnue_weights is empty.
 *
 * onnx_bytes is shared (shared_ptr to const) so that listeners may retain a reference
 * without copying the buffer.
 */
struct ModelBundle {
  std::shared_ptr<const std::vector<char>> onnx_bytes;
  std::map<std::string, std::vector<float>> nnue_weights;
};

/*
 * Parse the raw ONNX bytes received over the wire into a ModelBundle.
 *
 * Always populates ModelBundle::onnx_bytes (sharing ownership of the passed-in buffer).
 * Walks graph.initializer for tensors whose name starts with "nnue/", decoding each into a
 * std::vector<float> and inserting into nnue_weights under the prefix-stripped name. Only
 * FLOAT-typed tensors are supported (other dtypes throw).
 *
 * Returns true on successful protobuf parse, false otherwise. Even on parse failure,
 * onnx_bytes is set so that downstream consumers (e.g. TensorRT) can still attempt their
 * own parse and surface a more informative error.
 */
bool parse_model_bundle(std::shared_ptr<const std::vector<char>> raw, ModelBundle& out);

}  // namespace core
