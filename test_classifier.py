from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd

# get the 53 medicine class names from metadata
df = pd.read_csv("metadata.csv")
class_names = sorted(df["Class_Label"].unique().tolist())
print(f"Loaded {len(class_names)} classes")

# load CLIP (downloads once, then cached locally)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# test on one known image
test_image_path = r"E:\Anudip\pharmacy-vision-agent\raw\Acemiz\Image_2_jpg.rf.2dc375b802da8bea9fbd1c19b453790a.jpg"  
image = Image.open(test_image_path)

inputs = processor(text=class_names, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

top5_idx = probs[0].argsort(descending=True)[:5]
print("\nTop 5 predictions:")
for idx in top5_idx:
    print(f"{class_names[idx]}: {probs[0][idx].item():.2%}")