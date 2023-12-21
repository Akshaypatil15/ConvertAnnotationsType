# Script to convert yolo annotations to pascal voc format

from lxml.etree import tostring, fromstring
from xml.etree import cElementTree as ET
from cv2 import imread as ImageReader
from glob import glob
import os

CLASSES_COUNT = 1
CLASS_MAPPING = {str(a):str(a) for a in range(CLASSES_COUNT)}
EXPECTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

IMAGE_DIR_PREFIX = "/home/appy/project/kie_training/demo_data/images/"
ANNOTATIONS_DIR_PREFIX = "/home/appy/project/kie_training/demo_data/text_detection/"
DESTINATION_DIR = "/home/appy/project/kie_training/demo_data/xml_annotations/"


def get_class_name(lstr_key, ldict_clasess_mapping, default_class=None):
    if lstr_key in ldict_clasess_mapping.keys():
        return ldict_clasess_mapping.get(lstr_key)
    else:
        if default_class:
            return default_class
        return str(lstr_key)

def create_root(img_file_path, image_shape):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = os.path.basename(img_file_path)
    ET.SubElement(root, "folder").text = os.path.dirname(img_file_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_shape[0])
    ET.SubElement(size, "height").text = str(image_shape[1])
    ET.SubElement(size, "depth").text = str(image_shape[2])
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(img_file_path, save_dir, image_shape, voc_labels):
    root = create_root(img_file_path, image_shape)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root).getroot()
    lobj_lxml = fromstring(ET.tostring(tree, encoding='unicode'))
    file_prefix = '.'.join(os.path.basename(img_file_path).split(".")[:-1])
    with open(f"{save_dir}/{file_prefix}.xml", 'w') as fileWriter:
        fileWriter.write(tostring(lobj_lxml, pretty_print=True, encoding=str))


def read_file(img_path, label_dir, save_dir, classes_dict):
    file_prefix = '.'.join(os.path.basename(img_path).split(".")[:-1])
    label_file_path = os.path.join(label_dir, f"{file_prefix}.txt")

    if not os.path.exists(label_file_path):
        print(f"Missing annotation file. Skipped {os.path.basename(img_path)}...")
        return

    h, w, c = ImageReader(img_path).shape
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(get_class_name(data[0], classes_dict, "text"))
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc.append(int(center_x - round(bbox_width / 2)))
            voc.append(int(center_y - round(bbox_height / 2)))
            voc.append(int(center_x + round(bbox_width / 2)))
            voc.append(int(center_y + round(bbox_height / 2)))
            voc_labels.append(voc)
        create_file(img_path, save_dir, [w, h, c], voc_labels)
    print("Processing complete for file: {}".format(img_path))


if __name__ == "__main__":
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)

    for file_path in glob(IMAGE_DIR_PREFIX+"*.*"):
        if True in [os.path.basename(file_path).lower().endswith(img_extn) for img_extn in EXPECTED_IMAGE_EXTENSIONS]:
            read_file(file_path, ANNOTATIONS_DIR_PREFIX, DESTINATION_DIR, CLASS_MAPPING)