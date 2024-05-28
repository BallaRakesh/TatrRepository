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

class GroundTruthXML:
    
    def __init__(self,filename,Width,Height):
        self.root = ET.Element('GroundTruth',name=filename,width=Width,height=Height)
        Tables=ET.SubElement(self.root,'Tables')
        GroundTruth=ET.SubElement(Tables,'GroundTruth')
        self.tables=ET.SubElement(GroundTruth,'Tables')
        self.current_table = None
    
    def add_table(self,xmin_val,ymin_val,xmax_val,ymax_val):
        table=ET.SubElement(self.tables,'Table',x0=xmin_val,y0=ymin_val,x1=xmax_val,y1=ymax_val,orientation='unknown')
        self.current_table=table

    def add_row(self,xmin_val,ymin_val,xmax_val,ymax_val):
        row=ET.SubElement(self.current_table,'Row',x0=xmin_val,y0=ymin_val,x1=xmax_val,y1=ymax_val)
    
    def add_column(self,xmin_val,ymin_val,xmax_val,ymax_val):
        column=ET.SubElement(self.current_table,'Column',x0=xmin_val,y0=ymin_val,x1=xmax_val,y1=ymax_val)

    def add_cell(self,xmin_val,ymin_val,xmax_val,ymax_val,start_row,end_row,start_col,end_col,dont_care):
        cell=ET.SubElement(self.current_table,'Cell',x0=xmin_val,y0=ymin_val,x1=xmax_val,y1=ymax_val,startRow=start_row,endRow=end_row,startCol=start_col,endCol=end_col,dontCare=dont_care)
        cell.text='(0,0,0)'
    
    def save_xml(self,output_path):
        indent(self.root)
        tree = ET.ElementTree(self.root)
        tree.write(output_path)
        