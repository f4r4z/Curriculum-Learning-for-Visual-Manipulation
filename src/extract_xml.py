import xml.etree.ElementTree as ET
import os
from pathlib import Path
from pkg_resources import resource_filename
import numpy as np

def find_geoms_for_site(xml_path, site_name):
    # Parse the XML file to find geoms
    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_site = None
    for site in root.findall(".//site"):
        if site.get('name') == site_name:
            target_site = site
    
    geom_positions = []
    for body in root.find("./worldbody").findall(".//body"):
        for site in body.findall("./site"):
            if site.get('name') == site_name:
                for geom in body.findall("./geom"):
                    if geom.get("pos") is None:
                        continue
                    else:
                        geom_positions.append(np.fromstring(geom.get("pos"), dtype=np.float64, sep=' '))

    return geom_positions

def find_body_main(xml_path, site_name):
    # parse XML to find body name
    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_site = None
    for site in root.findall(".//site"):
        if site.get('name') == site_name:
            target_site = site
    
    geom_positions = []
    for body in root.find("./worldbody").findall(".//body"):
        for site in body.findall("./site"):
            if site.get('name') == site_name:
                return body.get('name')

    return None


def locate_libero_xml(object_name):
    file_name = object_name + ".xml"
    asset_dir = resource_filename("libero.libero", "")

    for dirpath, dirnames, filenames in os.walk(asset_dir):
        if (file_name) in filenames:
            return os.path.join(dirpath, file_name)


# path = locate_libero_xml("wooden_cabinet")
# print(find_geoms_for_site(path, "bottom_region"))
# print(find_body_main(path, "bottom_region"))