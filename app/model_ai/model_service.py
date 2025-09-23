import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image
import numpy as np
import cv2
import base64
import os
import random
import string
import json
from flask import Flask, request, jsonify
'''
class_names = [
    "Caries", "Crown", "Filling", "Implant", "Malaligned",
    "Mandibular Canal", "Missing teeth", "Periapical lesion",
    "Retained root", "Root Canal Treatment", "Root Piece",
    "Impacted Tooth", "maxillary sinus", "Bone Loss",
    "Fracture teeth", "Permanent Teeth", "Supra Eruption",
    "TAD", "abutment", "attrition", "bone defect",
    "gingival former", "metal band", "orthodontic brackets",
    "permanent retainer", "post - core", "plating", "wire",
    "Cyst", "Root resorption", "Primary teeth"
]
'''
class_names = [
    "Caries", "Crown", "Filling", "Implant",
    "Mandibular Canal", "Missing teeth", "Periapical lesion",
    "Root Canal Treatment", "Root Piece",
    "Impacted Tooth", "maxillary sinus", "Bone Loss",
    "post - core", "wire",
]

# Class descriptions
class_descriptions = {
    "Caries": "Tooth decay or cavities caused by bacterial activity that damages tooth enamel.",
    "Crown": "A dental restoration that completely caps or encircles a tooth or dental implant.",
    "Filling": "Material used to restore the function and morphology of missing tooth structure.",
    "Implant": "A surgical component that interfaces with the bone of the jaw to support a dental prosthesis.",
    "Mandibular Canal": "The canal within the mandible that contains the inferior alveolar nerve and blood vessels.",
    "Missing teeth": "Absence of one or more teeth in the dental arch.",
    "Periapical lesion": "Pathological changes in the bone around the apex of a tooth root, often due to infection.",
    "Root Canal Treatment": "A treatment sequence for the infected pulp of a tooth which results in the elimination of infection.",
    "Root Piece": "Fragment of a tooth root remaining in the jaw after tooth extraction or fracture.",
    "Impacted Tooth": "A tooth that has failed to erupt fully into its expected position.",
    "maxillary sinus": "The air-filled space located within the maxilla bone, above the teeth.",
    "Bone Loss": "Reduction in bone density and volume around teeth, often due to periodontal disease.",
    "post - core": "A foundation restoration for a tooth requiring a crown when there's insufficient tooth structure.",
    "wire": "Orthodontic wire used in braces to guide tooth movement and alignment.",
}
class_type = {
    "Caries": "Disease",
    "Periapical lesion": "Disease", 
    "Bone Loss": "Disease",
    "Impacted Tooth": "Condition",
    "Missing teeth": "Condition",
    "Root Piece": "Condition",
    "Mandibular Canal": "Anatomy",
    "maxillary sinus": "Anatomy",
    "Crown": "Previous Dental Work",
    "Filling": "Previous Dental Work",
    "Implant": "Previous Dental Work", 
    "Root Canal Treatment": "Previous Dental Work",
    "post - core": "Previous Dental Work",
    "wire": "Previous Dental Work"
}

np.random.seed(42)
colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(class_names))}

##MODEL_PATH = os.path.join(os.path.dirname(__file__), "maskrcnn_exportedLast.pth")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "maskrcnn_best.pth")
num_classes = len(class_names) + 1
##model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

# Create Mask R-CNN with ResNet-101 backbone
def create_maskrcnn_resnet101(num_classes):
    backbone = resnet_fpn_backbone('resnet101', pretrained=False)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
model = create_maskrcnn_resnet101(num_classes)
# Use ResNet-101 instead of ResNet-50
if os.path.exists(MODEL_PATH):
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    # Extract only the model weights from the checkpoint
    # The key might be "model", "state_dict", or something else depending on how it was saved
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        # If the checkpoint is just the model weights directly
        model_state_dict = checkpoint
    
    # Load only the model weights
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model loaded successfully")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

transform = T.Compose([T.ToTensor()])

def generate_random_id(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def run_inference(pil_image):
    if not os.path.exists(MODEL_PATH):
        return demo_inference(pil_image)
    
    image_tensor = transform(pil_image)
    with torch.no_grad():
        outputs = model([image_tensor])

    img = np.array(pil_image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    detections = []
    detection_data = []
    
    detection_ids = [generate_random_id() for _ in range(len(outputs[0]["boxes"]))]

    for i in range(len(outputs[0]["boxes"])):
        score = outputs[0]["scores"][i].item()
        if score > 0.5:
            box = outputs[0]["boxes"][i].cpu().numpy().astype(int)
            label = outputs[0]["labels"][i].item()
            mask = outputs[0]["masks"][i, 0].cpu().numpy()

            detection_id = detection_ids[i]
            class_name = class_names[label-1]
            color = colors[label-1]
            description = class_descriptions.get(class_name, "No description available.")
            types = class_type.get(class_name, "Unkown.")
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            label_text = f"ID:{detection_id} {class_name} {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            text_x = box[0]
            text_y = max(box[1] - 5, text_height + 5)
            
            cv2.rectangle(img, 
                         (text_x, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         color, -1)
            
            cv2.putText(img, label_text,
                       (text_x + 2, text_y - 2),
                       font, font_scale, (255, 255, 255), thickness)

            mask = outputs[0]["masks"][i, 0].cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8) * 255

            colored_mask = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 4), dtype=np.uint8)
            colored_mask[:, :, 0] = color[0]  # R
            colored_mask[:, :, 1] = color[1]  # G
            colored_mask[:, :, 2] = color[2]  # B
            colored_mask[:, :, 3] = (mask_binary > 0).astype(np.uint8) * 128  # Alpha channel

            mask_resized = cv2.resize(colored_mask, (pil_image.width, pil_image.height))

            _, mask_png = cv2.imencode(".png", mask_resized)
            mask_base64 = base64.b64encode(mask_png).decode("utf-8")

            detection_data.append({
                "id": detection_id,
                "class": class_name,
                "description": description,
                "types": types,
                "confidence": float(score),
                "bbox": box.tolist(),
                "color": color,
                "mask": mask_base64
            })
            
            detections.append({
                "id": detection_id,
                "class": class_name,
                "description": description,
                "types": types,
                "confidence": float(score),
                "bbox": box.tolist()
            })

    _, buffer = cv2.imencode(".jpg", img)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return detections, base64_image, detection_data

