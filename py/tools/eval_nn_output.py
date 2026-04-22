#!/usr/bin/env python3
import sys
import argparse
import onnxruntime as ort
import numpy as np

def print_all_output_shapes(onnx_path, out=None):
    # Logs to both the terminal AND the file
    def log(*a, **kw):
        print(*a, **kw)
        if out:
            print(*a, **kw, file=out)

    # Logs strictly to the file (bypassing the terminal)
    def file_only(*a, **kw):
        if out:
            print(*a, **kw, file=out)

    def log_tensor(name, arr):
        # Shape goes everywhere
        file_only(f"Head '{name}': {arr.shape}")

        # Stats and massive value arrays go ONLY to the file
        file_only(f"  min: {arr.min():.6g}, max: {arr.max():.6g}, mean: {arr.mean():.6g}, std: {arr.std():.6g}")
        file_only(f"  values:\n{np.array2string(arr, threshold=sys.maxsize, separator=', ')}\n")

    log(f"Loading {onnx_path}...")
    session = ort.InferenceSession(onnx_path)

    meta = session.get_modelmeta()
    log("--- Model Metadata ---")
    log(f"  Producer: {meta.producer_name}")
    log(f"  Version: {meta.version}")

    if meta.custom_metadata_map:
        log("  Custom Properties:")
        for k, v in meta.custom_metadata_map.items():
            log(f"    {k}: {v}")
    else:
        log("  (No custom metadata found)")
    log("")

    # 1. Figure out what the input expects
    input_node = session.get_inputs()[0]
    input_name = input_node.name
    input_shape = input_node.shape

    # If the batch size is dynamic (a string like 'batch'), set it to 1
    if isinstance(input_shape[0], str):
        input_shape[0] = 1

    # 2. Create a dummy input tensor filled with random noise
    log(f"Feeding dummy input of shape: {input_shape}")
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 3. Gather ALL output names from the model
    output_metadata = session.get_outputs()
    output_names = [output.name for output in output_metadata]
    log(f"Found {len(output_names)} output heads: {output_names}\n")

    # 4. Run inference for all outputs at once
    try:
        # By passing the full list of output_names, ONNX Runtime evaluates everything
        outputs = session.run(output_names, {input_name: dummy_input})

        # Print all output head shapes at the top
        log("--- Output Head Shapes ---")
        for name, out_array in zip(output_names, outputs):
            log(f"  {name}: {out_array.shape}")
        log("")  # Blank line for separation

        if out:
            file_only("--- Exact Output Values and Stats ---")

        # outputs is a list of numpy arrays returned in the same order as output_names
        for name, out_array in zip(output_names, outputs):
            log_tensor(name, out_array)

    except Exception as e:
        log(f"Error running inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output shapes of an ONNX model.")
    parser.add_argument("--input", '-i', required=True, help="Path to the ONNX model file")
    parser.add_argument("--output", '-o', help="Optional file path to write output to")
    args = parser.parse_args()

    if args.output:
        with open(args.output, 'w') as f:
            print_all_output_shapes(args.input, out=f)
    else:
        print_all_output_shapes(args.input)
