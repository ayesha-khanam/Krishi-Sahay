import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights

# Use modern weights (no warning)
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

categories = weights.meta["categories"]

transform = weights.transforms()

# keywords that indicate plant/leaf-like content in ImageNet labels
PLANT_KEYWORDS = [
    "plant", "leaf", "tree", "flower", "vegetable", "fruit", "corn", "wheat", "mushroom",
    "pepper", "tomato", "potato", "cucumber", "pumpkin", "banana", "pineapple", "orange",
    "strawberry", "grape", "pomegranate", "sunflower", "rose"
]

# Demo disease labels
DISEASE_CLASSES = [
    "Healthy Leaf",
    "Possible Fungal Infection",
    "Possible Nutrient Deficiency",
    "Possible Leaf Curl / Virus",
    "Possible Blight"
]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)

    label = categories[int(idx.item())].lower()
    confidence = float(conf.item())

    # Decide if image looks like plant/leaf based on label keywords
    is_leaf = any(k in label for k in PLANT_KEYWORDS)

    # Map to demo disease label (prototype)
    disease_label = DISEASE_CLASSES[int(idx.item()) % len(DISEASE_CLASSES)]

    return {
        "is_leaf": is_leaf,
        "imagenet_label": categories[int(idx.item())],
        "confidence": confidence,
        "disease_label": disease_label
    }