#!/usr/bin/env python3
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = Net().eval()
    example_input = torch.randn(1, 3).float()  # or float32 if you’ll do FP32

    torch.onnx.export(
        model,
        example_input,
        "model.onnx",
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        # **Here**: tell ONNX that dim 0 is dynamic ("batch")
        dynamic_axes={"input": {0: "batch"},
                      "output": {0: "batch"}}
    )
    print("Exported model.onnx")
