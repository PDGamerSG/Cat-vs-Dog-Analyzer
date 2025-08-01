import os, io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image as PILImage
from modal import App, Image
MODEL_FILENAME    = "cat_dog_model.pth"
HOST_MODEL_PATH   = "C:\Data\Coding Programs\Project\cat_dog_model.pth"
CONTAINER_PATH    = f"/{MODEL_FILENAME}"
app = App("cat-dog-app")
modal_image = (
    Image.debian_slim()
         .pip_install(["torch", "torchvision", "Pillow"])
         .add_local_file(HOST_MODEL_PATH, CONTAINER_PATH, copy=True)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
checkpoint = HOST_MODEL_PATH if os.path.exists(HOST_MODEL_PATH) else CONTAINER_PATH
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])
LABELS = ["cats", "dogs"]
@app.function(image=modal_image, timeout=300)
def classify_image(image_bytes: bytes) -> dict:
    img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    batch  = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)[0]
        probs  = F.softmax(logits, dim=0)
    return {
        "cats_prob": float(probs[0]*100),
        "dogs_prob": float(probs[1]*100),
        "label": LABELS[int(probs[1] > probs[0])],
    }

if __name__ == "__main__":
    app.deploy()
