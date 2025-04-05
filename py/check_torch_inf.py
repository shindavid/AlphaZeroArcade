import os
import torch
import glob
import re

def contains_inf(tensor):
    """Check if the tensor contains any infinite values."""
    is_all_finite = torch.isfinite(tensor).all().item()
    return not is_all_finite

def check_model_file(file_path):
    """
    Load the model from the given file and check for infinite values
    in its parameters.

        bool: True if any parameter contains inf, False otherwise.
    """
    try:
        # Attempt to load the state_dict
        state_dict = torch.load(file_path, map_location='cpu')

        # If the loaded object is a state_dict, it should be a dict
        if isinstance(state_dict, dict):
            has_inf = False
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if contains_inf(tensor):
                        print(f"  Inf detected in parameter: {key}")
                        has_inf = True
            return has_inf
        else:
            # If it's not a state_dict, it might be the whole model
            # Attempt to iterate through its parameters
            has_inf = False
            for name, param in state_dict.named_parameters():
                if contains_inf(param.data):
                    print(f"  Inf detected in parameter: {name}")
                    has_inf = True
            return has_inf
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return False

# Function to extract gen number
def get_gen_number(filename):
    pattern = re.compile(r'gen-(\d+)\.pt$')
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # Assign a default value if pattern not found

def main():
    # Specify the directory containing the .pt files
    models_dir = "/workspace/repo/output/c4/main_160/models"

    # Use glob to find all .pt files in the directory
    pattern = os.path.join(models_dir, "gen-*.pt")
    model_files = glob.glob(pattern)
    model_files = sorted(model_files, key=get_gen_number)

    if not model_files:
        print(f"No .pt files found in directory: {models_dir}")
        return

    print(f"Checking {len(model_files)} model files for infinite values...\n")

    for file_path in model_files:
        file_name = os.path.basename(file_path)
        print(f"Checking {file_name}...")
        has_inf = check_model_file(file_path)
        if has_inf:
            print(f"  [WARNING] {file_name} contains infinite values.\n")
        # else:
            # print(f"  [OK] {file_name} does not contain infinite values.\n")

if __name__ == "__main__":
    main()
