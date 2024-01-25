import os
os.system('git clone https://github.com/facebookresearch/detectron2.git')
os.system('pip install -e detectron2')
import sys
sys.path.append("detectron2")
from unilm.dit.object_detection.ditod import add_vit_config
import torch
import cv2
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import gradio as gr


cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("cascade_dit_base.yml")

cfg.MODEL.WEIGHTS = "publaynet_dit-b_cascade.pth"

cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)


def analyze_image(img):
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    if cfg.DATASETS.TEST[0] == 'icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

    output = predictor(img)["instances"]

    # Filter instances to keep only those corresponding to tables
    table_instances = output[output.pred_classes == md.thing_classes.index("table")]

    v = Visualizer(img[:, :, ::-1],
                   md,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)

    # Draw instance predictions for tables only
    result = v.draw_instance_predictions(table_instances.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # Get bounding box details
    bbox_details = []
    for i in range(len(table_instances)):
        instance = table_instances[i]
        bbox = instance.pred_boxes.tensor.cpu().numpy().tolist()
        score = instance.scores.cpu().numpy().item()
        bbox_details.append({"bbox": bbox, "score": score})

    return result_image, bbox_details


title = " Table Detection with DiT"
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

iface = gr.Interface(
    fn=analyze_image,
    inputs=[gr.Image(type="numpy", label="document image")],
    outputs=[gr.Image(type="numpy", label="detected tables"), gr.JSON(label="bounding box details")],
    title=title,

    css=css,
)
iface.launch(debug=True, share=True)
