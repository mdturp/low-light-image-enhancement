# %%
# VSCODE Interactive Notebook
# Transform the model to ONNX format and test it.

# %%
from model import Finetunemodel
from PIL import Image
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
import torchvision.transforms as transforms

# %%
# Define helper functions for image processing.


def load_images_transform(file):
    # Load and transform the image
    transform = transforms.Compose([transforms.ToTensor()])
    im = Image.open(file).convert("RGB")
    img_tensor = transform(im)

    # Convert to numpy and add a batch dimension
    img_numpy = img_tensor.numpy().astype(np.float32)
    img_batch = np.expand_dims(img_numpy, 0)

    return torch.from_numpy(img_batch)


def save_images(tensor, path):
    # Convert tensor to a numpy array and rescale
    image_numpy = tensor.detach().numpy()[0]
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype("uint8")

    # Save the image
    Image.fromarray(image_numpy).save(path, "png")


def display_images(tensor):
    # Convert tensor to a numpy array and rescale
    image_numpy = tensor.detach().numpy()[0]
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype("uint8")

    # Display the image
    Image.fromarray(image_numpy).show()


# %%
# Load the model and test that it works correctly.
model = Finetunemodel("./model_weights_difficult.pt")
model.eval()

img = load_images_transform("./test.jpg")
out = model(img)
save_images(out, "./test_pytorch_output.png")

# %%
# Convert the model to ONNX format.
scripted_model = torch.jit.script(model)

dummy_input = torch.randn(1, 3, 1000, 1000)
dynamic_axes = {
    "input": {
        0: "batch_size",
        2: "height",
        3: "width",
    },  # Input layer with dynamic batch size, height, and width
    "output": {
        0: "batch_size",
        2: "height",
        3: "width",
    },  # Output layer with dynamic batch size, height, and width
}
torch.onnx.export(
    scripted_model,
    dummy_input,
    "model.onnx",
    export_params=True,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)

# %%
# Load the ONNX model into the ONNX runtime and test it.

ort_session = ort.InferenceSession("model.onnx")
outputs = ort_session.run(None, {"input": img.numpy()})
save_images(torch.from_numpy(outputs[0]), "test_onnx_output.png")

# %%
print("Completed successfully!")