import os, io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image as PILImage
from modal import App, Image

MODEL_FILENAME = "cat_dog_model.pth"
app = App("cat-dog-app")
modal_image = (
    Image.debian_slim()
         .pip_install(["torch", "torchvision", "Pillow"])
         .add_local_file(
             MODEL_FILENAME,
             f"/{MODEL_FILENAME}",
             copy=True
         )
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(f"{MODEL_FILENAME}", map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
LABELS = ["cats", "dogs"]
def classify_image_with_scores(image_bytes: bytes) -> dict:
    img    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    batch  = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)[0]
        probs  = F.softmax(logits, dim=0)
    all_scores = {}

    for i in range(len(LABELS)):
        label = LABELS[i]
        probability = float(probs[i])*100
        all_scores[label] = probability

    idx = int(torch.argmax(probs))
    return {
        "all_scores":      all_scores,
        "predicted_label": LABELS[idx],
    }
def classify_image_bytes(image_bytes: bytes) -> str:
    return classify_image_with_scores(image_bytes)["predicted_label"]

@app.function(image=modal_image, timeout=300)
def classify_image(image_bytes: bytes) -> str:
    return classify_image_bytes(image_bytes)

if __name__ == "__main__":
    app.deploy()
