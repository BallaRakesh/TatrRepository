import utils

def load_ocr():
    cfg = utils.load_config('config.ini')

    selected_ocr = 'tesseract'  # default
    for item in cfg['OCR']:
        if cfg['OCR'][item].lower() == 'true':
            selected_ocr = item
            break
    # #print(selected_ocr)
    # exit()


    credentials_path = cfg['PATH']['GV_CREDENTIALS']

    ocr_client = utils.get_ocr_client(selected_ocr, credentials_path)
    
    return selected_ocr, ocr_client
    
def apply(document_image):
    selected_ocr, ocr_client = load_ocr()
    ocr_data = utils.perform_ocr(ocr=selected_ocr, 
                                 ocr_client=ocr_client, 
                                 image_path = document_image)