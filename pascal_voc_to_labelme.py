from lxml import etree as ET
from glob import glob
import json

def xml_to_json(path):
    for xml_file in glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        main_dict = {
                'version': '5.0.1',
                'flags': {},
                'shapes': [],
                'imagePath': root.find('filename').text,
                'imageData': None,
                'imageHeight': int(root.find('size')[1].text),
                'imageWidth': int(root.find('size')[0].text)
                }

        for member in root.findall('object'):
            xmin, ymin, xmax, ymax = [int(child.text) for child in member[4]]
            ldict_shape = {'label': member[0].text,
                           'points': [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                           'group_id': None, 'shape_type': 'polygon', 'flags': {}}
            main_dict['shapes'].append(ldict_shape)
        json_save_path = xml_file.replace(".xml", ".json")
        with open(json_save_path, "w") as outfile:
            json.dump(main_dict, outfile, indent = 4)

xml_to_json("sample_data")