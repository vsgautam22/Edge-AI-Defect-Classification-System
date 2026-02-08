import torch
import torch.nn as nn
from torchvision import models
import os

# ================= CONFIG =================
MODEL_PATH = "phase1_multiclass_model.pth"
ONNX_OUTPUT_PATH = "edge_ai_wafer_system.onnx"
NUM_CLASSES = 10
DEVICE = "cpu"   # ONNX export should be done on CPU

# ================= LOAD MODEL =================
print("Loading model...")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully")

# ================= DUMMY INPUT =================
# This matches your training input: (B, C, H, W)
dummy_input = torch.randn(1, 3, 224, 224)

# ================= EXPORT TO ONNX =================
print("Exporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"ONNX export completed: {ONNX_OUTPUT_PATH}")
print("Project Done by Team ChipCrafters.....Precision in Every Pixel!!!!.....")
