import os
from glob import glob
from lxml import etree as ET
from PIL import Image


def get_image_size(image_file_name):
    return Image.open(image_file_name).shape

def create_root(input_dir_path, file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = input_dir_path
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
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

def create_file(input_file_path, output_dir, width, height, voc_labels):
    input_dir_path = input_file_path.replace(os.path.basename(input_file_path), "")
    file_prefix = os.path.basename(input_file_path).replace(".txt", "")
    xml_file_name = os.path.join(output_dir,"{}.xml".format(file_prefix))

    root = create_root(input_dir_path, file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write(xml_file_name, pretty_print=True)

def read_file(input_file_path, output_dir, image_shape, class_mapping):
    # input_dir_path = input_file_path.replace(os.path.basename(input_file_path), "")
    # file_prefix = os.path.basename(input_file_path).replace(".txt", "")

    w, h, _ = image_shape
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            # voc.append(class_mapping.get(data[0]))
            voc.append("0")
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc.append(int(center_x - round(bbox_width / 2)))
            voc.append(int(center_y - round(bbox_height / 2)))
            voc.append(int(center_x + round(bbox_width / 2)))
            voc.append(int(center_y + round(bbox_height / 2)))
            voc_labels.append(voc)
        create_file(input_file_path, output_dir, w, h, voc_labels)
    print("Processing complete for file: {}".format(input_file_path))

def start(IMAGE_DIR, LABELS_DIR, OUTPUT_DIR, CLASS_MAPPING):
    for dir_name in ["image_files"]:
        current_dir = os.path.join(IMAGE_DIR, dir_name)
        save_dir = os.path.join(OUTPUT_DIR, dir_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for filename in glob(current_dir+"/*.txt"):
            if not os.path.basename(filename) == 'classes.txt':
                read_file(filename, save_dir, get_image_size(), CLASS_MAPPING)
            else:
                print("Skipping file: {}".format(os.path.basename(filename)))

if __name__ == "__main__":
    images_dir = ""
    labels_dir = ""
    output_dir = ""
    # start(ANNOTATIONS_DIR_PREFIX, DESTINATION_DIR, CLASS_MAPPING)