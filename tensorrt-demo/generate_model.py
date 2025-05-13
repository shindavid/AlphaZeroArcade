#!/usr/bin/env python3
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # simple linear layer: 3 → 2
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = Net().eval()

    # dummy input: batch=1, 3 features
    example_input = torch.randn(1, 3)

    torch.onnx.export(
        model,
        example_input,
        "model.onnx",
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={"input": {0: "batch"}}  # allow runtime batch change
    )
    print("Exported model.onnx")
