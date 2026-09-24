import gradio as gr
from classify_image import load_model, classify_image, transform
from stock_manager import load_stock, check_availability, add_to_reorder_list, get_reorder_list
from PIL import Image
import tempfile
import os

model, class_names = load_model()
stock_df = load_stock()

CONFIDENCE_THRESHOLD = 0.60
HIGH_CONFIDENCE_CUTOFF = 0.85  # above this, don't bother showing runner-up matches

STATUS_COLORS = {
    "In stock": "#16a34a",
    "Low stock": "#d97706",
    "Out of stock": "#dc2626",
}

CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}
#title-block {
    text-align: center;
    padding: 10px 0 20px 0;
}
#result-card {
    border-radius: 12px;
    padding: 20px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
    font-size: 0.9em;
}
"""


def identify_and_check(image):
    if image is None:
        return "Please upload a photo.", None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    results = classify_image(tmp_path, model, class_names, top_k=3)
    os.remove(tmp_path)

    top_name, top_confidence = results[0]

    if top_confidence < CONFIDENCE_THRESHOLD:
        html = f"""
        <div id='result-card'>
            <h3>⚠️ Not confident this matches a known medicine</h3>
            <p>Best guess: <b>{top_name}</b> — only {top_confidence:.1%} confidence</p>
            <p>This may be a product outside our current 53 trained medicines.</p>
            <p><b>Other close matches:</b></p>
            <ul>
                {''.join(f"<li>{name} ({conf:.1%})</li>" for name, conf in results[1:])}
            </ul>
        </div>
        """
        return html, None

    availability = check_availability(top_name, stock_df)
    color = STATUS_COLORS.get(availability["status"], "#64748b")

    if top_confidence >= HIGH_CONFIDENCE_CUTOFF:
        other_matches_html = ""
    else:
        other_matches_html = f"""
        <p><b>Other possible matches:</b></p>
        <ul>
            {''.join(f"<li>{name} ({conf:.1%})</li>" for name, conf in results[1:])}
        </ul>
        """

    html = f"""
    <div id='result-card'>
        <h2>💊 {top_name}</h2>
        <p style='font-size: 1.1em;'>Confidence: <b>{top_confidence:.1%}</b></p>
        <p>
            <span class='status-badge' style='background:{color};'>
                {availability['status']}
            </span>
            &nbsp; {availability['quantity']} units in stock
        </p>
        {other_matches_html}
    </div>
    """
    return html, top_name


def handle_reorder(medicine_name):
    if not medicine_name:
        return "⚠️ No medicine identified yet."
    return "✅ " + add_to_reorder_list(medicine_name)


def view_reorder_list():
    df = get_reorder_list()
    if df.empty:
        return "Reorder list is empty."
    return df.to_string(index=False)


theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Pharmacy Medicine Identifier", theme=theme, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="title-block"):
        gr.Markdown("# 💊 Pharmacy Medicine Identifier")
        gr.Markdown("Upload a photo of a medicine to identify it and check stock availability.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload medicine photo", height=280)
            identify_btn = gr.Button("🔍 Identify & Check Stock", variant="primary", size="lg")

            result_output = gr.HTML()
            identified_name = gr.State()

            with gr.Row():
                reorder_btn = gr.Button("➕ Add to Reorder List")
            reorder_output = gr.Textbox(label="Reorder status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Current Reorder List")
            view_btn = gr.Button("🔄 Refresh Reorder List")
            reorder_list_output = gr.Textbox(label="Reorder List", interactive=False, lines=18)

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