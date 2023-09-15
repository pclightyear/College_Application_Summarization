# Utility variable
import sys
sys.path.insert(0, '../..')

import var.path as P
import os

def get_application_pdf_file(_year, _id):
    _year = str(_year)
    _id = str(_id)
    return os.path.join(P.FP_FULL_APPLICATION_DIR, _year, P.PDF_DIR, "{}.pdf".format(_id))
    
def get_application_page_img_dir(_year, _id):
    _year = str(_year)
    _id = str(_id)
    return os.path.join(P.FP_FULL_APPLICATION_DIR, _year, P.IMAGE_DIR, _id)
    
def get_application_page_img_file(_year, _id, _pn):
    _year = str(_year)
    _id = str(_id)
    _pn = str(_pn)
    return os.path.join(P.FP_FULL_APPLICATION_DIR, _year, P.IMAGE_DIR, _id, "{}.jpg".format(_pn))
    
def get_application_page_raw_ocr_dir(_year, _id):
    _year = str(_year)
    _id = str(_id)
    return os.path.join(P.FP_FULL_APPLICATION_DIR, _year, P.TXT_OCR_RAW_DIR, _id) 
   
def get_application_ocr_file(_year, _id):
    _year = str(_year)
    _id = str(_id)
    return os.path.join(P.FP_FULL_APPLICATION_DIR, _year, P.TXT_OCR_DIR, "{}.json".format(_id))

