import gradio as gr
from classify_image import load_model, classify_image, transform
from stock_manager import load_stock, check_availability, add_to_reorder_list, get_reorder_list
from PIL import Image
import tempfile
import os

model, class_names = load_model()
stock_df = load_stock()

CONFIDENCE_THRESHOLD = 0.60  # below this, treat as "unknown / not confident"


def identify_and_check(image):
    if image is None:
        return "Please upload a photo.", None

    # save the uploaded PIL image to a temp file, since classify_image expects a path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    results = classify_image(tmp_path, model, class_names, top_k=3)
    os.remove(tmp_path)

    top_name, top_confidence = results[0]

    if top_confidence < CONFIDENCE_THRESHOLD:
        output_lines = [
            f"⚠️ Not confident this matches a known medicine (best guess: {top_name}, only {top_confidence:.1%} confidence).",
            "",
            "This may be a product outside our current 53 trained medicines. Other close matches:",
        ]
        for name, conf in results[1:]:
            output_lines.append(f"- {name} ({conf:.1%})")
        return "\n".join(output_lines), None  # None means no valid name to reorder

    availability = check_availability(top_name, stock_df)

    output_lines = [
        f"**Identified: {top_name}** ({top_confidence:.1%} confidence)",
        "",
        f"Stock status: **{availability['status']}** ({availability['quantity']} units)",
        "",
        "Other possible matches:",
    ]
    for name, conf in results[1:]:
        output_lines.append(f"- {name} ({conf:.1%})")

    return "\n".join(output_lines), top_name


def handle_reorder(medicine_name):
    if not medicine_name:
        return "No medicine identified yet."
    return add_to_reorder_list(medicine_name)


def view_reorder_list():
    df = get_reorder_list()
    if df.empty:
        return "Reorder list is empty."
    return df.to_string(index=False)


with gr.Blocks(title="Pharmacy Medicine Identifier") as demo:
    gr.Markdown("# Pharmacy Medicine Identifier & Stock Checker")
    gr.Markdown("Upload a photo of a medicine to identify it and check stock availability.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload medicine photo")
            identify_btn = gr.Button("Identify & Check Stock", variant="primary")
            result_output = gr.Markdown()
            identified_name = gr.State()

            reorder_btn = gr.Button("Add to Reorder List")
            reorder_output = gr.Textbox(label="Reorder status", interactive=False)

        with gr.Column():
            gr.Markdown("### Current Reorder List")
            view_btn = gr.Button("Refresh Reorder List")
            reorder_list_output = gr.Textbox(label="Reorder list", interactive=False, lines=15)

    identify_btn.click(
        identify_and_check,
        inputs=image_input,
        outputs=[result_output, identified_name],
    )
    reorder_btn.click(
        handle_reorder,
        inputs=identified_name,
        outputs=reorder_output,
    )
    view_btn.click(
        view_reorder_list,
        outputs=reorder_list_output,
    )

if __name__ == "__main__":
    demo.launch()