# Pharmacy Medicine Identifier & Stock Checker

Upload a photo of a medicine, get it identified by a fine-tuned image classifier, check simulated stock availability, and add out-of-stock or low-stock items to a persistent reorder list.

## What It Does

1. Upload a photo of a medicine (box, blister pack, etc.)
2. A fine-tuned MobileNetV2 model identifies which of 53 known medicines it most closely matches
3. The app checks a stock file and reports availability: In stock / Low stock / Out of stock
4. If needed, add the medicine to a reorder list, saved persistently to a CSV file

## Architecture

1. **Model training** (`train_classifier.py`) — transfer learning on a pretrained MobileNetV2, fine-tuned on ~5,000 labeled medicine images across 53 classes (Pharmassist dataset, Mendeley Data). Achieved 96.8% validation accuracy after 5 epochs.
2. **Inference** (`classify_image.py`) — loads the trained model and classifies a new image, returning the top-k most likely matches with confidence scores.
3. **Stock management** (`stock_manager.py`) — checks a medicine's stock status against a data file, and manages a persistent reorder list.
4. **Interface** (`app.py`) — a Gradio app tying it together: photo upload → identification → stock check → optional reorder.

## Why a Fine-Tuned Model, Not Zero-Shot

An earlier attempt used CLIP for zero-shot classification (no training required) but performed poorly — the correct medicine often wasn't even in its top 5 guesses. This makes sense: CLIP was trained on general web images and captions, and has no prior knowledge of specific, unfamiliar medicine brand names. Fine-tuning a small classifier directly on labeled examples of these exact products performed dramatically better (96.8% validation accuracy vs. CLIP's unreliable guesses).

## Important Limitation: Closed-Set Classification

**This model can only recognize the 53 specific medicines it was trained on.** When shown a genuinely new photo (different background, lighting, and — critically — a product outside the trained set), it will still confidently output one of its 53 known classes, because that is structurally all it can predict. It has no built-in way to say "I don't know this product."

In real-world testing, a photo of **Paracip-500** (not one of the 53 trained products) was misclassified as **Dolo-650** with 71.4% confidence — a confident, wrong answer. A confidence threshold (currently 60%) is used to flag genuinely uncertain predictions, but it cannot catch cases where the model is confidently wrong rather than uncertain. This is a known, structural limitation of closed-set image classifiers, not a bug — expanding the training set to include more products is the only real fix, and even then, the model will always be limited to whatever it has explicitly been trained on.

## Data Sources

- **Images**: "Pharmassist Dataset," Mendeley Data — ~5,000 images across 53 medicine classes, raw and augmented versions, all photographed indoors under LED lighting. Not included in this repo due to size (~1.8GB); download separately from Mendeley Data if you want to retrain.
- **Stock quantities**: Synthetic/simulated data generated for this project (`create_stock_data.py`) — not real pharmacy inventory. Quantities are randomly assigned for demonstration purposes.

## Technology Stack

Python · PyTorch · torchvision · Pandas · Gradio

## Running Locally

The trained model (`medicine_classifier.pt`) is included in this repo, so you can run the app directly without retraining:

```bash
uv venv
.venv\Scripts\activate
uv pip install torch torchvision pandas gradio pillow
```

Generate simulated stock data (if `stock_data.csv` isn't already present):
```bash
python create_stock_data.py
```

Run the app:
```bash
python app.py
```

To retrain the model from scratch, download the Pharmassist Dataset from Mendeley Data, place the `raw` and `augmented` folders in the project root, and run:
```bash
python train_classifier.py
```

## Known Limitations

- Closed-set classifier: only recognizes the 53 medicines it was trained on; will confidently misclassify anything else (see above).
- Stock data is synthetic, not real pharmacy inventory.
- Training images were all captured indoors under consistent LED lighting; performance on very different lighting/backgrounds is untested beyond the one real-world example described above.
- A confidence threshold helps flag uncertain predictions but does not catch confidently wrong ones.

## Author

Ramya Murugesh