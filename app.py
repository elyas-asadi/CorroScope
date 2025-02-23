import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import json
from pydantic import BaseModel, Field
from functools import lru_cache
import hashlib

sys.path.append('.')

# TransUNet imports
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# Initialize OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o"

class Data_output(BaseModel):
    rating: int = Field(
        ..., 
        description="Rate the corrosion severity from 0 (no corrosion) to 10 (severe corrosion)"
    )
    description: str = Field(
        ..., 
        description="Brief description of the corrosion observed (max 50 words)"
    )
    confidence: float = Field(
        ..., 
        description="The confidence score (from 0 to 1) representing the certainty of the analysis"
    )
    recommendation: str = Field(
        ...,
        description="Brief, specific recommendation for addressing the corrosion (max 20 words)"
    )

# Model initialization
IMAGE_SIZE = 640
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
config_vit.n_classes = 3  # Changed from 2 to 3 to match your training
global_model = ViT_seg(config_vit, img_size=640, num_classes=3)
global_model.load_state_dict(torch.load("./models/statdict.pth"))
global_model = global_model.to(device)
global_model.eval()

class AnalysisState:
    def __init__(self):
        self.masks = []
        self.analyses = []
        self.llm_analyses = []
        self.selected_index = None
        self.cache = {}

state = AnalysisState()

def get_file_info(file):
    """Get formatted file size and dimensions"""
    size_mb = os.path.getsize(file.name) / (1024 * 1024)
    try:
        with Image.open(file.name) as img:
            dimensions = f"{img.width}x{img.height}"
        return f"{dimensions} ({size_mb:.1f} MB)"
    except:
        return f"({size_mb:.1f} MB)"

def add_text_overlay(img, text, image_number):
    """Add text overlay to image with complete information"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    
    complete_text = f"Image {image_number}: {text}"
    
    text_bbox = draw.textbbox((0, 0), complete_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    margin = 15
    draw.rectangle(
        [(margin, margin), 
         (text_width + margin*2, text_height + margin*2)],
        fill=(0, 0, 0, 180)
    )
    
    draw.text(
        (margin*2, margin*2),
        complete_text,
        fill=(255, 255, 255),
        font=font
    )
    
    return img

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_image_hash(image):
    """Calculate a hash for the image to use as cache key"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return hashlib.md5(buffered.getvalue()).hexdigest()

def resize_for_llm(image):
    """Resize image to width of 256px while maintaining aspect ratio"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    target_width = 256
    aspect_ratio = image.size[1] / image.size[0]
    target_height = int(target_width * aspect_ratio)
    
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_image

@lru_cache(maxsize=100)
def cached_analyze_with_llm(image_hash):
    """Cached version of LLM analysis"""
    if image_hash in state.cache:
        return state.cache[image_hash]
    return None

def batch_analyze_with_llm(images, batch_size=3):
    """Analyze multiple images in batches"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = []
        
        for img in batch:
            img_hash = calculate_image_hash(img)
            cached_result = cached_analyze_with_llm(img_hash)
            
            if cached_result:
                batch_results.append(cached_result)
            else:
                try:
                    result = analyze_single_image_llm(img)
                    state.cache[img_hash] = result
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error in LLM analysis: {str(e)}")
                    batch_results.append(json.dumps({
                        "rating": 0,
                        "description": f"Error in LLM analysis: {str(e)}",
                        "confidence": 0.0,
                        "recommendation": ""
                    }))
        
        results.extend(batch_results)
    
    return results

def analyze_single_image_llm(image):
    """Analyze single image with LLM"""
    resized_image = resize_for_llm(image)
    base64_image = image_to_base64(resized_image)
    
    system_prompt = """
    You are an AI assistant specializing in analyzing images of structural steel corrosion. Follow these instructions:

    1. Examine the provided image carefully for signs of corrosion, focusing on:
    - Surface condition and texture
    - Color variations and patterns
    - Presence of rust, pitting, or material loss
    - Paint or coating condition (if present)
    - Structural elements and joint conditions

    2. Rate the severity of corrosion on a scale from 1 to 10:
    - 1: Pristine condition (no visible corrosion, intact mill scale/coating)
    - 2: Very minor surface rust (<1% surface affected, no material loss)
    - 3: Early stage corrosion (1-5% surface affected, superficial rust)
    - 4: Developing corrosion (5-15% surface affected, initial pattern formation)
    - 5: Moderate corrosion (15-25% surface affected, clear pattern development)
    - 6: Progressive corrosion (25-35% surface affected, early pitting)
    - 7: Advanced corrosion (35-45% surface affected, defined pitting/scaling)
    - 8: Severe corrosion (45-60% surface affected, material loss evident)
    - 9: Critical corrosion (60-75% surface affected, significant deterioration)
    - 10: Extreme corrosion (>75% surface affected, severe structural concern)

    3. Provide a short, specific recommendation for addressing the corrosion.
    """
    
    try:
        content = [
            {
                "type": "text",
                "text": "Analyze this image for corrosion"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.3,
            response_format=Data_output
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"LLM Analysis failed: {str(e)}")

# Initialize transform
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def process_image(image_path):
    """Process single image for model input with better error handling"""
    try:
        # Read image with OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is numpy array before resize
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
            
        # Resize with explicit interpolation method
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply transformations
        transformed = transform(image=image)
        return transformed['image'].unsqueeze(0)
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        # Return None instead of raising to handle in caller
        return None




def get_segmentation_mask(image_tensor):
    """Get segmentation mask using global model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = global_model(image_tensor)
        # Take the second channel for corrosion probability
        mask_prob = torch.sigmoid(output[0, 1]).cpu().numpy()
        # The third channel could be used for confidence if needed
        confidence = torch.sigmoid(output[0, 2]).cpu().numpy()
        mask = (mask_prob > 0.5).astype(np.uint8) * 255
        return mask, confidence

def get_severity_circle(severity_score):
    """Get colored circle based on severity level (1-10)"""
    # Ensure the score is between 1 and 10
    severity_score = max(1, min(10, float(severity_score)))
    
    severity_circles = {
        1: "🟢",  # No/minimal corrosion
        2: "🟢",
        3: "🟡",  # Early stage
        4: "🟡",
        5: "🟠",  # Moderate
        6: "🟠",
        7: "🟤",  # Advanced
        8: "🟤",
        9: "🔴",  # Critical
        10: "🔴"  # Extreme corrosion
    }
    
    # Round to nearest integer for circle selection
    return severity_circles[round(severity_score)]


def process_images(files, use_llm=False, progress=gr.Progress()):
    """Process images with robust error handling"""
    if files is None or len(files) == 0:
        return [], None, "Please upload some images first."
    
    images = []
    file_info = []
    masks = []
    analyses = []
    llm_analyses = []
    
    progress(0, desc="Initializing...")
    total_files = len(files)
    total_steps = 2 if use_llm else 1
    
    # Prepare images for batch processing if LLM is enabled
    if use_llm:
        llm_images = []
    
    for idx, file in enumerate(files):
        try:
            progress((idx + 1) / (total_files * total_steps), 
                    desc=f"Processing image {idx + 1}/{total_files}")
            
            # Process image for display
            img = Image.open(file.name)
            if img is None:
                raise ValueError(f"Failed to open image {file.name}")
                
            info = get_file_info(file)
            filename = os.path.basename(file.name)
            
            if use_llm:
                llm_images.append(img)
            
            # Add text overlay to display image
            overlay_text = f"{filename} {info}"
            img_with_text = add_text_overlay(img.copy(), overlay_text, idx + 1)
            images.append(img_with_text)
            file_info.append(info)
            
            # Process image for model
            image_tensor = process_image(file.name)
            if image_tensor is None:
                continue
                
            # Get segmentation mask and confidence
            mask, confidence = get_segmentation_mask(image_tensor)
            
            # Resize mask and confidence to match image dimensions
            h, w = np.array(img).shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Create colored mask for visualization
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            colored_mask[:] = (83, 110, 127)  # Background color #263743 in BGR

            # Create corrosion overlay
            corrosion_mask = np.zeros_like(colored_mask)
            corrosion_mask[mask > 127] = (100, 43, 57)  # Coral color #FF7453 in BGR format

            # Apply confidence weighting only to corrosion areas
            weighted_corrosion = (corrosion_mask.astype(float) * confidence[:, :, np.newaxis]).astype(np.uint8)

            # Combine background with weighted corrosion
            mask_condition = (mask > 127)[:, :, np.newaxis]  # Expand mask to match 3 channels
            final_mask = np.where(mask_condition, weighted_corrosion, colored_mask)
            masks.append(final_mask)

            # Calculate corrosion percentage weighted by confidence
            weighted_mask = (mask > 127).astype(float) * confidence
            corrosion_percentage = np.sum(weighted_mask) / np.sum(confidence) * 100
            
            # Calculate severity score (1-10)
            if corrosion_percentage == 0:
                seg_severity = 1
            else:
                seg_severity = min(1 + (corrosion_percentage / 8.5), 10)
            
            seg_severity_circle = get_severity_circle(seg_severity)
            
            # Initialize analysis text with confidence information
            avg_confidence = np.mean(confidence)
            analysis = f"""### Image {idx + 1}: {filename} {info}

**Segmentation Analysis**:
- **Severity Level**: {seg_severity_circle} {corrosion_percentage:.1f}% affected area | Rating: {int(seg_severity)}/10

- **Status**: Automated corrosion detection completed"""
            
            analyses.append(analysis)
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            analyses.append(f"### Image {idx + 1}: {filename}\nError: Failed to process image")
            continue

    # Perform batch LLM analysis if enabled
    if use_llm and llm_images:
        try:
            progress(0.5, desc="Performing Contextual analysis...")
            llm_results = batch_analyze_with_llm(llm_images)
            
            for idx, (analysis, llm_result) in enumerate(zip(analyses, llm_results)):
                try:
                    if isinstance(llm_result, str):
                        result = json.loads(llm_result)
                    else:
                        result = llm_result
                        
                    llm_rating = result['rating']
                    llm_description = result['description']
                    llm_confidence = result['confidence']
                    llm_recommendation = result['recommendation']
                    
                    llm_severity_circle = get_severity_circle(llm_rating)
                    
                    # Add LLM analysis to the text, integrating confidence from segmentation
                    if "Error:" not in analysis:
                        seg_confidence = np.mean(confidence) if 'confidence' in locals() else 0.5
                        combined_confidence = (seg_confidence + llm_confidence) / 2
                        
                        analyses[idx] = analysis + f"""

**Contextual Analysis**:
- **Severity Level**: {llm_severity_circle} (Rating: {llm_rating}/10)
- **Description**: {llm_description}
- **Combined Confidence**: {combined_confidence:.2f}
- **Recommendation**: {llm_recommendation}"""
                    
                except Exception as e:
                    print(f"Error parsing LLM result for image {idx}: {str(e)}")
                    if "Error:" not in analyses[idx]:
                        analyses[idx] = analysis + "\n\n**Error**: Failed to process LLM analysis"
                        
        except Exception as e:
            print(f"Error in batch LLM analysis: {str(e)}")
            for idx, analysis in enumerate(analyses):
                if "Error:" not in analysis:
                    analyses[idx] = analysis + "\n\n**Error**: Failed to perform LLM analysis"

    if not images:
        return [], None, "Error processing images. Please try again."
    
    # Store in state
    state.masks = masks
    state.analyses = analyses
    state.llm_analyses = llm_analyses if use_llm else []
    
    # Return the processed results with safe indexing
    return (
        images, 
        state.masks[0] if state.masks else None, 
        state.analyses[0] if state.analyses else "No analysis available"
    )


    ##################
def on_select(evt: gr.SelectData):
    """Handle image selection"""
    selected_idx = evt.index
    state.selected_index = selected_idx
    
    if 0 <= selected_idx < len(state.masks):
        return state.masks[selected_idx], state.analyses[selected_idx]
    return None, "No analysis available for this image"

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("""# CorroScope: Advanced Corrosion Analysis System
        Combining computer vision and large language modles for comprehensive corrosion assessment.""")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Images")
                gr.Markdown("Maximum recommended batch: 10 images")
                with gr.Row():
                    image_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Drop images here or click to upload",
                        elem_id="file_upload"
                    )
               
        # Add custom CSS for better button styling
        gr.HTML("""
            <style>
                
            /* Progress bar and timer text styling */
            .progress-text, 
            [class*="progress-text"],
            .progress-level,
            [class*="progress-level"],
            .timer-text,
            [class*="timer-text"],
            .progress-bar ~ div {
                font-size: 1.2rem !important;
                font-weight: 300 !important;
            }
                
            .custom-button-fast {
                background-color: #C9C7DB !important;  /* Your yellow color */
                border-color: #C9C7DB !important;
            }
            .custom-button-fast:hover {
                background-color: #C9C7DB !important;  /* Slightly darker shade for hover */
                border-color: #C9C7DB !important;
            }
            .custom-button-slow {
                background-color: #E8E7BE !important;  /* Your purple color */
                border-color: #E8E7BE !important;
            }
            .custom-button-slow:hover {
                background-color: #E8E7BE !important;  /* Slightly darker shade for hover */
                border-color: #E8E7BE !important;
            }
            .option-container {
                border: 1px solid #e6e9ef;
                border-radius: 8px;
                padding: 20px;
                margin: 8px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .option-title {
                font-weight: 600;
                font-size: 1.1em;
                margin-bottom: 12px;
                text-align: center;
                color: #374151;
            }
            .option-description {
                color: #6b7280;
                font-size: 0.9em;
                margin: 12px 0;
                text-align: center;
            }
            .processing-time {
                background: #f9fafb;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 0.85em;
                color: #6b7280;
                margin-top: 12px;
                text-align: center;
                border: 1px solid #f3f4f6;
            }
            .clear-container {
                display: flex;
                justify-content: center;
                padding: 16px 0;
            }
            </style>
        """)
        
        # Add guidance text
        gr.Markdown("### After uploading images, please select one of the options below:")
        
        # Processing options layout
        with gr.Row():
            # Basic processing option
            with gr.Column(scale=5):
                with gr.Column(elem_classes="option-container"):
                    gr.Markdown("### Quick Analysis", elem_classes="option-title")
                    gr.Markdown("Segmentation-only processing for rapid results", elem_classes="option-description")
                    process_basic_btn = gr.Button(
                        "🔄 Process with segmentation model only", 
                        elem_classes="custom-button-fast",
                        size="lg"
                    )
                    gr.Markdown("⚡ Processing time: <~1 seconds per image", elem_classes="processing-time")
            
            # Advanced processing option
            with gr.Column(scale=5):
                with gr.Column(elem_classes="option-container"):
                    gr.Markdown("### Detailed Analysis", elem_classes="option-title")
                    gr.Markdown("Complete analysis with AI-powered insights", elem_classes="option-description")
                    process_advanced_btn = gr.Button(
                        "🤖 Process with segmentation and contextual analysis", 
                        elem_classes="custom-button-slow",
                        size="lg"
                    )
                    gr.Markdown("⏳ Processing time: ~5-10 seconds per image", elem_classes="processing-time")
        
        # Centered clear button
        with gr.Row():
            with gr.Column(elem_classes="clear-container"):
                clear_btn = gr.Button(
                    "🗑️ Clear All", 
                    variant="stop",
                    size="lg"
                )
        
        with gr.Row():
            status_text = gr.Markdown("Ready to process images...")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original Images")
                image_gallery = gr.Gallery(
                    label="Original Images",
                    show_label=False,
                    columns=4,
                    height=400,
                    allow_preview=True,
                    preview=True,
                    elem_id="image_gallery"
                )
            
            with gr.Column():
                gr.Markdown("### Corrosion Map")
                mask_image = gr.Image(
                    label="Corrosion Detection Mask",
                    show_label=False,
                    height=400,
                    elem_id="mask_image"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Analysis Results")
                text_output = gr.Markdown(
                    elem_id="text_output",
                    value="Upload and process images to see analysis..."
                )
        
        # Event handlers
        def update_gallery(files):
            if not files:
                return None, None, "Ready to process images..."
            num_files = len(files)
            return (
                [file.name for file in files],
                None,
                f"✨ Loaded {num_files} images. Select an analysis option to begin."
            )
        
        image_input.change(
            fn=update_gallery,
            inputs=[image_input],
            outputs=[image_gallery, mask_image, status_text]
        )
        
        # Handler for basic processing (segmentation only)
        process_basic_btn.click(
            fn=lambda: "⏳ Processing images with quick analysis...",
            outputs=status_text
        ).then(
            fn=lambda x: process_images(x, use_llm=False),
            inputs=[image_input],
            outputs=[image_gallery, mask_image, text_output]
        ).then(
            fn=lambda: "✅ Quick analysis complete!",
            outputs=status_text
        )
        
        # Handler for advanced processing (with LLM)
        process_advanced_btn.click(
            fn=lambda: "⏳ Processing images with detailed AI analysis...",
            outputs=status_text
        ).then(
            fn=lambda x: process_images(x, use_llm=True),
            inputs=[image_input],
            outputs=[image_gallery, mask_image, text_output]
        ).then(
            fn=lambda: "✅ Detailed analysis complete!",
            outputs=status_text
        )
        
        clear_btn.click(
            fn=lambda: (None, None, None, "Ready to process images..."),
            outputs=[image_input, mask_image, text_output, status_text]
        )

        image_gallery.select(
            fn=on_select,
            outputs=[mask_image, text_output]
        )

        return interface
    
if __name__ == "__main__":
    # Test model with random input
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    output = global_model(input_tensor)
    print("Model output shape:", output.shape)
    
    # Launch the interface
    demo = create_interface()
    demo.launch()