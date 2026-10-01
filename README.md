# Pharmacy Medicine Identifier & Stock Checker

Upload a photo of a medicine, get it identified by a fine-tuned image classifier, check simulated stock availability, and add out-of-stock or low-stock items to a persistent reorder list.

## What It Does

1. Upload a photo of a medicine (box, blister pack, etc.)
2. A fine-tuned MobileNetV2 model identifies which of 53 known medicines it most closely matches
3. The app checks a stock file and reports availability: In stock / Low stock / Out of stock
4. If needed, add the medicine to a reorder list, saved persistently to a CSV file

## Architecture

1. **Model training** (`train_classifier.py`) — transfer learning on a pretrained MobileNetV2, fine-tuned on ~5,000 labeled medicine images across 53 classes (Pharmassist dataset, Mendeley Data).
2. **Inference** (`classify_image.py`) — loads the trained model and classifies a new image, returning the top-k most likely matches with confidence scores.
3. **Stock management** (`stock_manager.py`) — checks a medicine's stock status against a data file, and manages a persistent reorder list.
4. **Interface** (`app.py`) — a Gradio app tying it together: photo upload → identification → stock check → optional reorder.

## Why a Fine-Tuned Model, Not Zero-Shot

An earlier attempt used CLIP for zero-shot classification (no training required) but performed poorly — the correct medicine often wasn't even in its top 5 guesses. This makes sense: CLIP was trained on general web images and captions, and has no prior knowledge of specific, unfamiliar medicine brand names. Fine-tuning a small classifier directly on labeled examples of these exact products performed dramatically better.

## Important Limitations

### Closed-Set Classification
This model can only recognize the 53 specific medicines it was trained on. When shown a product outside that set, it will still confidently output one of its 53 known classes, because that is structurally all it can predict. It has no built-in way to say "I don't know this product" on its own — the app adds a confidence threshold to flag likely-unknown cases, described further below.

A photo of **Paracip-500** (not one of the 53 trained products) was tested against two versions of the model: the original model misclassified it as Dolo-650 (71.4% confidence); after retraining with added augmentation, it was misclassified differently, as Emeset (78.4% confidence) — a different wrong answer, at even higher confidence. This shows that retraining changed *what* the model got wrong without fixing *that* it was wrong — a useful, honest finding about the limits of this kind of fix.

### Real-World Generalization Gap
Beyond unknown products, the model's confidence and accuracy vary depending on how a *known* medicine is photographed:

- **Shelcal 500**, photographed as a blister pack, was correctly identified at 93.3% confidence. The same product, photographed as its outer box, was only 32.0% confident and the top guess was wrong — correctly flagged as low-confidence by the app.
- **Sompraz**, a genuinely trained class, was confidently (94.4%) misclassified as "Pan 20" in a real-world test photo, with no warning shown.
- **Dolo-650**, photographed in a real-world setting, was correctly identified at 80.9% confidence — confirming the model does work reliably in some real-world conditions, not only on dataset images.

### An Attempt to Improve Generalization
A second training run added stronger data augmentation (random crops, rotation, color jitter, perspective distortion, blur) and unfroze the last several layers of the pretrained backbone, instead of only training the final classification layer. This raised validation accuracy from 96.8% to 99.4% — but validation accuracy alone was not a reliable predictor of real-world performance. Some cases improved (Shelcal box-angle confidence rose from 15.6% to 32.0%, now correctly flagged as uncertain rather than silently wrong). Others did not meaningfully improve (Paracip-500 remained confidently wrong, just with a different wrong label), and new failures surfaced during broader testing (Sompraz).

**Honest takeaway**: closing the gap between clean training-distribution images and messy real-world photos is a genuinely hard problem, and a single round of augmentation/fine-tuning tweaks did not reliably solve it. A more thorough fix would likely require a larger, more deliberately diverse training set — including real-world photos taken under varied, uncontrolled conditions similar to how an end user would actually use this tool.

### Where It Does Work Well
Out of five real-world test photos: two were correctly identified with reasonable confidence (Dolo-650 at 80.9%, and an earlier Shelcal 500 blister-pack photo at 93.3%). One case (Sompraz) was confidently wrong with no warning. The remaining two cases involved low-confidence flags from the app:

- **Shelcal 500**, photographed as its outer box (a framing not well represented in training), triggered a low-confidence warning (32.0%) — correctly signaling uncertainty, even though the underlying product was actually known.
- **Selotol XL 50**, a genuinely unknown medicine (not one of the 53 trained classes), also triggered the app's low-confidence warning (41.0%), which explicitly states the product may be outside the trained set — correctly describing the actual situation, not just a vague uncertainty flag.

This is a realistic, mixed picture: the confidence threshold catches some but not all failure cases, and correctly flags genuine unknowns at least some of the time — a partial, not complete, safety net.

## Data Sources

- **Images**: "Pharmassist Dataset," Mendeley Data — real-world photographs of used tablet blister packs, collected from personal and voluntary sources across Chennai, India (September–December 2025), under varying angles, orientations, lighting, and backgrounds. 2,050 original photographs plus 2,947 augmented versions (4,997 images total) across 53 medicine classes. Not included in this repo due to size (~1.8GB); download separately from [Mendeley Data](https://data.mendeley.com/datasets/zbnhmbbymd/1) if you want to retrain.
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

- Closed-set classifier: only recognizes the 53 medicines it was trained on; will confidently misclassify anything else.
- Real-world photos of *known* medicines can still be misclassified, sometimes confidently, when photographed differently than the training data's framing (see above).
- Stock data is synthetic, not real pharmacy inventory.
- A confidence threshold helps flag uncertain predictions but does not catch confidently wrong ones — it caught some genuine failures (Shelcal box angle, Selotol XL 50) but missed others (Sompraz).
- A round of added augmentation and partial fine-tuning showed mixed results — better calibration in some cases, new or persisting confident failures in others. Improving real-world generalization remains an open problem for this project.

## Author

Ramya Murugesh