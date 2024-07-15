import utils
from columns.detect import detect_column
from columns.merge import get_corrected_columns
from copy import deepcopy

def detect_columns(header_res, image, ocr_data, single_line_elements):
    # col_data, header_cells, image_h, header_strip, ocr_data, single_line_elements

    # col_image  = deepcopy(image)
    # if header_res['final_header'] != None:
    #     col_image[0:header_res['final_header'][1], 0:header_res['final_header'][2]] = 255
    #     table_coords = [header_res['final_header'][0], header_res['final_header'][1], 
    #                     col_image.shape[1], col_image.shape[0]]
    #     ocr_data = utils.remove_words(table_coords=table_coords, table_ocr=ocr_data)

    cv_columns = detect_column(ocr_data, image)
    
    image_h, image_w = image.shape[:2]
    ocr_data = list(ocr_data.values())
    
    for i in range(len(single_line_elements)):
        single_line_elements[i] = single_line_elements[i].lower().strip()
    
    print(cv_columns)
    
    # print(f"\n\n\n-------------------------{header_res['element_coordinate']}\n\n\n")
    final_columns_headers = get_corrected_columns(
        col_data=cv_columns,
        header_cells=header_res['element_coordinate'],
        image_h=image_h,
        header_strip=header_res['final_header'],
        ocr_data=ocr_data,
        single_line_elements=single_line_elements,
        header_dict = header_res['header_dict']
    )
    
    return final_columns_headers


