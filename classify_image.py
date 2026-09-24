import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def load_model(model_path="medicine_classifier.pt"):
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_image(image_path, model, class_names, top_k=5):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idx = probs.topk(top_k)
    results = [(class_names[i], p.item()) for p, i in zip(top_probs, top_idx)]
    return results


if __name__ == "__main__":
    model, class_names = load_model()

    # test on a known image — adjust path to a real one from your raw folder
    test_image = r"E:\Anudip\pharmacy-vision-agent\raw\Acemiz\Image_2_jpg.rf.2dc375b802da8bea9fbd1c19b453790a.jpg" 
    results = classify_image(test_image, model, class_names)
    print(f"Testing image: {test_image}\n")
    for name, prob in results:
        print(f"{name}: {prob:.2%}")