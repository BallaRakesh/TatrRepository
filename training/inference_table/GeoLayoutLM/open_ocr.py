from mmocr.apis import MMOCRInferencer
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
ocr('dataset/funsd_geo/testing_data/images/82092117.png', show=True, print_result=True)