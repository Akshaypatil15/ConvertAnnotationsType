"""
Single Inference => 29/29 [00:56<00:00,  1.96s/it]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from mmocr.apis import init_detector
from mmocr.utils.model import revert_sync_batchnorm

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.apis.inference import model_inference

from glob import glob
from tqdm import tqdm
import json
import cv2


INPUT_DIR = "sample_data/images/"
CLASSES_FILE = "sample_data/classes.txt"
EXPECTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

with open(CLASSES_FILE, 'r') as file_reader:
    llst_classes = file_reader.readlines()
LDICT_CLASSES = {llst_classes[i].strip(): i for i in range(len(llst_classes))}

# Build recognition model
recog_config = "models/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py"
recog_ckpt = "models/sar_r31_parallel_decoder_academic-dba3a4a3.pth"
recog_model = init_detector( recog_config, recog_ckpt, device='cpu')
recog_model = revert_sync_batchnorm(recog_model)

def read_labelme_annotation(lstr_file_path):
    with open(lstr_file_path) as file_reader:
        return json.load(file_reader)

def text_ocr(llst_box_coordinates, image_array):
    box = [round(i) for i in llst_box_coordinates]
    box_img = crop_img(image_array, box)
    return  model_inference(recog_model, box_img, batch_mode=False)["text"]

def get_annotations(llst_shapes, ldict_classes, larr_image):
    box = [b for a in llst_shapes['points'] for b in a]
    return {"box": box, "text": text_ocr(box, larr_image), "label": ldict_classes.get(llst_shapes['label'])}

def convert_into_kie_format(ldict_input, ldict_classes, img_path):
    ldict_kie_format = {"file_name": img_path,
                        "height": ldict_input['imageHeight'], "width": ldict_input['imageWidth'],
                        "annotations": []}

    larr_image = cv2.imread(img_path)
    for shapes in tqdm(ldict_input['shapes']):
        ldict_kie_format["annotations"].append(get_annotations(shapes, ldict_classes, larr_image.copy()))
    return ldict_kie_format

llst_test_data = []
for img_path in glob(INPUT_DIR+"/*.*"):
    if True in [os.path.basename(img_path).lower().endswith(img_extn) for img_extn in EXPECTED_IMAGE_EXTENSIONS]:
        file_prefix = '.'.join(os.path.basename(img_path).split(".")[:-1])

        lstr_json_path = os.path.join(INPUT_DIR, f"{file_prefix}.json")
        if not os.path.exists(lstr_json_path):
            print(f"No JSON Found. Skipping {os.path.basename(img_path)}...")
            continue

        ldict_labelme_annotation = read_labelme_annotation(lstr_json_path)
        ldict_kie_format = convert_into_kie_format(ldict_labelme_annotation, LDICT_CLASSES, img_path)

        llst_test_data.append(ldict_kie_format)

# Saving json file
with open("sample_data/test.text", "w") as outfile:
    outfile.write(str(llst_test_data))