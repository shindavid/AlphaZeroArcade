#!/usr/bin/env python3
import argparse
import onnx
from onnx import helper, TensorProto
import numpy as np
import subprocess
import sys

"""
Converts a Leela Chess Zero (Lc0) neural network to ONNX format and performs surgery so a0achess
binary could use it.

This script orchestrates a two-step pipeline to prepare an Lc0 model (typically a .pb.gz
file) for specific inference environments. First, it utilizes an external `lc0` binary to
convert the model into an intermediate ONNX graph (opset 17). Second, it uses the ONNX API
to apply structural modifications to the model's inputs, outputs, and metadata.

Specific Graph Modifications (Surgery):
    1. Input Renaming: Renames the primary input tensor to 'input' and updates dependent nodes.
    2. Head Pruning: Removes the '/output/mlh' (Moves Left Head) output completely.
    3. Head Injection: Injects a dummy 'action_value' output head (shape [1, 1858, 2])
       populated with a constant float value. Defaults to -0.693147 (approx. ln(0.5)).
    4. Metadata Tagging: Appends the property 'model-architecture-signature'='lc0-neural-net'.

Example Usage:
    python modify_lc0_onnx.py \
        --input=/workspace/repo/extra_deps/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
        --output=/workspace/repo/extra_deps/lc0/BT4.onnx \
        --lc0-path=/workspace/repo/extra_deps/lc0/lc0
"""
def leela2onnx(input_path, output_path, lc0_path, batch_size=-1):
    print(f"Loading pb.gz model from {input_path}...")

    command = [
        lc0_path,
        "leela2onnx",
        f"--input={input_path}",
        f"--output={output_path}",
        "--onnx-opset=17"
    ]
    if batch_size > 0:
        command.append(f"--batch-size={batch_size}")

    print(f"Executing command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        print("\nConversion successful!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Conversion failed with exit code {e.returncode}.", file=sys.stderr)
        print(f"Error details:\n{e.stderr}", file=sys.stderr)

    except FileNotFoundError:
        print(f"\n[Error] Could not find the lc0 executable at: {lc0_path}", file=sys.stderr)
        print("Please check the path and ensure the file has execute permissions.", file=sys.stderr)


def rename_graph_input(model, new_input_name="input"):
    """Renames the first input of the model and updates all dependent nodes."""
    if not model.graph.input:
        return

    old_input_name = model.graph.input[0].name
    print(f"Renaming input from '{old_input_name}' to '{new_input_name}'...")

    model.graph.input[0].name = new_input_name

    # Update all nodes that used the old input name
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old_input_name:
                node.input[i] = new_input_name


def remove_graph_output(model, target_name):
    """Removes a specific output from the model graph by name."""
    print(f"Removing the '{target_name}' head...")
    filtered_outputs = [out for out in model.graph.output if out.name != target_name]
    model.graph.ClearField("output")
    model.graph.output.extend(filtered_outputs)


def inject_constant_output(model, output_name, shape, value):
    """Injects a Constant node into the graph and exposes it as an output."""
    print(f"Preparing dummy tensor '{output_name}' with shape {shape} and value {value}...")

    # Create raw array and ONNX tensor
    dummy_array = np.full(shape, value, dtype=np.float32)
    tensor = helper.make_tensor(
        name=f"dummy_{output_name}_tensor",
        data_type=TensorProto.FLOAT,
        dims=shape,
        vals=dummy_array.flatten().tolist()
    )

    # Create the Constant node
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        value=tensor,
        name=f"ConstantInjector_{output_name}"
    )

    # Create the metadata to expose it as an official output
    output_value_info = helper.make_tensor_value_info(
        name=output_name,
        elem_type=TensorProto.FLOAT,
        shape=shape
    )

    # Append to graph
    model.graph.node.append(constant_node)
    model.graph.output.append(output_value_info)


def set_metadata_property(model, key, value):
    """Sets a metadata property in the ONNX model, overwriting if it exists."""
    print(f"Injecting metadata signature: {key}='{value}'...")

    # Keep everything except the key we are modifying to avoid duplicates
    existing_props = [p for p in model.metadata_props if p.key != key]
    model.ClearField("metadata_props")
    model.metadata_props.extend(existing_props)

    new_prop = model.metadata_props.add()
    new_prop.key = key
    new_prop.value = value


def modify_onnx_model(input_path, output_path, dummy_value):
    """Pipeline to load, apply surgeries, and save the ONNX model."""
    print(f"Loading original model from {input_path}...")
    model = onnx.load(input_path)

    rename_graph_input(model, new_input_name="input")
    remove_graph_output(model, target_name="/output/mlh")
    inject_constant_output(model, output_name="action_value", shape=[1, 1858, 2], value=dummy_value)
    set_metadata_property(model, key="model-architecture-signature", value="lc0-neural-net")

    print(f"Saving modified model to {output_path}...")
    onnx.save(model, output_path)
    print("Surgery complete! Model is ready for inference.")


def main():
    parser = argparse.ArgumentParser(description="Inject a dummy action_value head into an ONNX chess network.")
    parser.add_argument("--input", "-i", required=True, help="Path to the original ONNX model (e.g., BT4.onnx)")
    parser.add_argument("--output", "-o", required=True, help="Path to save the modified ONNX model")
    parser.add_argument("--lc0-path", required=True, help="Path to the lc0 executable for conversion")
    parser.add_argument("--value", "-v", type=float, default=-0.693147,
                        help="The constant float value to fill the tensor with (default: -0.693147 (ln(0.5)))")

    args = parser.parse_args()
    intermediate_onnx_path = args.output + ".raw"
    leela2onnx(args.input, intermediate_onnx_path, args.lc0_path)
    modify_onnx_model(intermediate_onnx_path, args.output, args.value)


if __name__ == "__main__":
    main()
