import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem       

def generate_xml(imgname: str, 
                 table_w: int, 
                 table_h: float, 
                 bboxes: list, 
                 labels: list,
                 img_extension: str = 'png'):
    
    
    if 'xml' in imgname:
        imgname = imgname.replace('xml',img_extension)
        
    root = ET.Element('annotation')
    
    tree = ET.ElementTree(root)
    
    # folder = ET.SubElement(root)
    filename = ET.SubElement(root, 'filename')
    filename.text = imgname
    
    source = ET.SubElement(root, 'source')
    db = ET.SubElement(source, 'database')
    db.text = 'table_extraction'
    annotation = ET.SubElement(source, 'annotation')
    annotation.text = 'Unknown'
    img = ET.SubElement(source, 'image')
    img.text = 'Unknown'
    
    img_size = ET.SubElement(root, 'size')
    img_width = ET.SubElement(img_size, 'width')
    img_width.text = str(table_w)
    img_height = ET.SubElement(img_size, 'height')
    img_height.text = str(table_h)
    img_depth = ET.SubElement(img_size, 'depth')
    
    segmented = ET.SubElement(source, 'segmented')
    segmented.text = '0'
        
    for i in range(len(bboxes)):
        object = ET.SubElement(root, 'object')
        
        obj_name = ET.SubElement(object, 'name')
        obj_name.text = labels[i]
        obj_truncated = ET.SubElement(object, 'truncated')
        obj_truncated.text = '0' 
        obj_occluded = ET.SubElement(object, 'occluded')
        obj_occluded.text = '0'
        obj_difficult = ET.SubElement(object, 'difficult')
        obj_difficult.text = '0'
        
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bboxes[i][0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bboxes[i][1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bboxes[i][2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bboxes[i][3])
    
    indent(root)
    return tree