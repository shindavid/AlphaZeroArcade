#!/usr/bin/env python3
import argparse
import onnx
from onnx import helper, TensorProto
import numpy as np

def inject_dummy_head(input_path, output_path, dummy_value):
    print(f"Loading original model from {input_path}...")
    model = onnx.load(input_path)

    # Renaming the input to "input" to satisfy the check in NeuralNet.cpp:check_model_compatibility()
    old_input_name = model.graph.input[0].name
    new_input_name = "input"
    print(f"Renaming input from '{old_input_name}' to '{new_input_name}'...")

    model.graph.input[0].name = new_input_name

    # Update all nodes that used the old input name
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old_input_name:
                node.input[i] = new_input_name

    print("Removing the '/output/mlh' head...")
    # Keep all outputs EXCEPT the one named '/output/mlh'
    filtered_outputs = [out for out in model.graph.output if out.name != '/output/mlh']
    model.graph.ClearField("output")
    model.graph.output.extend(filtered_outputs)

    # 1. Define the exact shape your engine requires
    shape = [1, 1858, 2]
    print(f"Preparing dummy tensor with shape {shape} and value {dummy_value}...")

    # 2. Create the raw data array
    dummy_array = np.full(shape, dummy_value, dtype=np.float32)

    # 3. Convert the numpy array into an ONNX Tensor object
    tensor = helper.make_tensor(
        name="dummy_action_value_tensor",
        data_type=TensorProto.FLOAT,
        dims=shape,
        vals=dummy_array.flatten().tolist()
    )

    # 4. Create a Constant node that generates this tensor out of thin air
    constant_node = helper.make_node(
        "Constant",
        inputs=[],                 # No inputs needed
        outputs=["action_value"],  # MUST exactly match the name your engine expects!
        value=tensor,
        name="ActionValueConstantInjector"
    )

    # 5. Create the metadata that tells the ONNX runtime this is an official output
    output_value_info = helper.make_tensor_value_info(
        name="action_value",
        elem_type=TensorProto.FLOAT,
        shape=shape
    )

    # 6. Append the new node to the graph's computation steps
    model.graph.node.append(constant_node)

    # 7. Append the output definition to the graph's exposed outputs
    model.graph.output.append(output_value_info)

    # 7.5. Inject the model architecture signature metadata
    # This satisfies the check in NeuralNet.cpp:set_model_architecture_signature()
    print("Injecting model architecture signature...")

    # Check if a property with this key already exists to avoid duplicates
    signature_key = "model-architecture-signature"
    signature_value = "lc0-neural-net"

    # Clear existing if necessary, then add
    existing_props = [p for p in model.metadata_props if p.key != signature_key]
    model.ClearField("metadata_props")
    model.metadata_props.extend(existing_props)

    new_prop = model.metadata_props.add()
    new_prop.key = signature_key
    new_prop.value = signature_value

    # 8. Save the modified model
    print(f"Saving modified model to {output_path}...")
    onnx.save(model, output_path)
    print("Surgery complete! Model is ready for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a dummy action_value head into an ONNX chess network.")
    parser.add_argument("--input", "-i", required=True, help="Path to the original ONNX model (e.g., BT4.onnx)")
    parser.add_argument("--output", "-o", required=True, help="Path to save the modified ONNX model")

    # We default to 0.0, but let you override it via the command line!
    parser.add_argument("--value", "-v", type=float, default=-0.693147,
                        help="The constant float value to fill the tensor with (default: -0.693147 (ln(0.5)))")

    args = parser.parse_args()

    inject_dummy_head(args.input, args.output, args.value)
