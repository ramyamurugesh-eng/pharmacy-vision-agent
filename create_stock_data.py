import pandas as pd
import random

# use the same class names the model was trained on
from classify_image import load_model

_, class_names = load_model()

random.seed(42)  # reproducible "random" stock numbers

stock_data = []
for name in class_names:
    quantity = random.randint(0, 100)  # includes some genuinely out-of-stock items
    stock_data.append({"MedicineName": name, "Quantity": quantity})

df = pd.DataFrame(stock_data)
df.to_csv("stock_data.csv", index=False)
print(df)
print(f"\nSaved {len(df)} medicines to stock_data.csv")