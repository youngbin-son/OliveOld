import os
import time
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
from IPython.display import HTML, FileLink, display

# Import local modules

utils_file_path = Path("./utils/notebook_utils.py")
notebook_directory_path = Path(".")

sys.path.append(str(utils_file_path.parent))
sys.path.append(str(notebook_directory_path))

from notebook_utils import load_image
from model.u2net import U2NET, U2NETP

"""### Settings
[back to top ⬆️](#Table-of-contents:)

This tutorial supports using the original U$^2$-Net salient object detection model, as well as the smaller U2NETP version. Two sets of weights are supported for the original model: salient object detection and human segmentation.
"""

model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    model=U2NETP,
    model_args=(),
)
u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)
u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP",
    model=U2NET,
    model_args=(3, 1),
)

# Set u2net_model to one of the three configurations listed above.
u2net_model = u2net_lite

# The filenames of the downloaded and converted models.
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")

"""### Load the U$^2$-Net Model
[back to top ⬆️](#Table-of-contents:)

The U$^2$-Net human segmentation model weights are stored on Google Drive. They will be downloaded if they are not present yet. The next cell loads the model and the pre-trained weights.
"""

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")

# Load the model.
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights.
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

"""## Convert PyTorch U$^2$-Net model to OpenVINO IR
[back to top ⬆️](#Table-of-contents:)

We use model conversion Python API to convert the Pytorch model to OpenVINO IR format.
Executing the following command may take a while.
"""

model_ir = ov.convert_model(net, example_input=torch.zeros((1,3,512,512)), input=([1, 3, 512, 512]))

"""## Load and Pre-Process Input Image
[back to top ⬆️](#Table-of-contents:)

While OpenCV reads images in `BGR` format, the OpenVINO IR model expects images in `RGB`. Therefore, convert the images to `RGB`, resize them to `512 x 512`, and transpose the dimensions to the format the OpenVINO IR model expects.

We add the mean values to the image tensor and scale the input with the standard deviation. It is called the input data normalization before propagating it through the network. The mean and standard deviation values can be found in the [dataloader](https://github.com/xuebinqin/U-2-Net/blob/master/data_loader.py) file in the [U^2-Net repository](https://github.com/xuebinqin/U-2-Net/) and multiplied by 255 to support images with pixel values from 0-255.
"""

IMAGE_URI = "./imgs/makeuped.jpg"

input_mean = np.array([123.675, 116.28 , 103.53]).reshape(1, 3, 1, 1)
input_scale = np.array([58.395, 57.12 , 57.375]).reshape(1, 3, 1, 1)

image = cv2.cvtColor(
    src=load_image(IMAGE_URI),
    code=cv2.COLOR_BGR2RGB,
)

resized_image = cv2.resize(src=image, dsize=(512, 512))
# Convert the image shape to a shape and a data type expected by the network
# for OpenVINO IR model: (1, 3, 512, 512).
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

input_image = (input_image - input_mean) / input_scale

"""## Select inference device
[back to top ⬆️](#Table-of-contents:)

select device from dropdown list for running inference using OpenVINO
"""

import ipywidgets as widgets

core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)


"""## Do Inference on OpenVINO IR Model
[back to top ⬆️](#Table-of-contents:)

Load the OpenVINO IR model to OpenVINO Runtime and do inference.
"""

core = ov.Core()
# Load the network to OpenVINO Runtime.
compiled_model_ir = core.compile_model(model=model_ir, device_name=device.value)
# Get the names of input and output layers.
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

# Do inference on the input image.
start_time = time.perf_counter()
result = compiled_model_ir([input_image])[output_layer_ir]
end_time = time.perf_counter()
print(
    f"Inference finished. Inference time: {end_time-start_time:.3f} seconds, "
    f"FPS: {1/(end_time-start_time):.2f}."
)

"""## Visualize Results
[back to top ⬆️](#Table-of-contents:)

Show the original image, the segmentation result, and the original image with the background removed.
"""

# Resize the network result to the image shape and round the values
# to 0 (background) and 1 (foreground).
# The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
resized_result = np.rint(
    cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))
).astype(np.uint8)

# Create a copy of the image and set all background values to 255 (white).
bg_removed_result = image.copy()
bg_removed_result[resized_result == 0] = 255

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
ax[0].imshow(image)
ax[1].imshow(resized_result, cmap="gray")
ax[2].imshow(bg_removed_result)
for a in ax:
    a.axis("off")

BACKGROUND_FILE = "./back/uni.jpg"
OUTPUT_DIR = "output"

os.makedirs(name=OUTPUT_DIR, exist_ok=True)

background_image = cv2.cvtColor(src=load_image(BACKGROUND_FILE), code=cv2.COLOR_BGR2RGB)
background_image = cv2.resize(src=background_image, dsize=(image.shape[1], image.shape[0]))

# Set all the foreground pixels from the result to 0
# in the background image and add the image with the background removed.
background_image[resized_result == 1] = 0
new_image = background_image + bg_removed_result

# Save the generated image.
new_image_path = Path(f"{OUTPUT_DIR}/{Path(IMAGE_URI).stem}-{Path(BACKGROUND_FILE).stem}.jpg")
cv2.imwrite(filename=str(new_image_path), img=cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

# Display the original image and the image with the new background side by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
ax[0].imshow(image)
ax[1].imshow(new_image)
for a in ax:
    a.axis("off")
plt.show()

# Create a link to download the image.
image_link = FileLink(new_image_path)
image_link.html_link_str = "<a href='%s' download>%s</a>"
display(
    HTML(
        f"The generated image <code>{new_image_path.name}</code> is saved in "
        f"the directory <code>{new_image_path.parent}</code>. You can also "
        "download the image by clicking on this link: "
        f"{image_link._repr_html_()}"
    )
)

