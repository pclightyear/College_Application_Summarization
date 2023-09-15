import json

# Utility variable
import sys
sys.path.insert(0, '../..')

import utils.get_path as GP
import utils.io as IO

from opencc import OpenCC
cc = OpenCC('s2tw')

def get_application_text(_year, _id, _print=False):
    fp = GP.get_application_ocr_file(_year, _id)

    with open(fp, 'r') as f:
        app_texts = json.load(f)
        p_app_texts = []
        
        if _print:
            print("Number of pages: {}".format(len(app_texts)))

        for pn, page_text in enumerate(app_texts, 1):
            p_app_texts.append(cc.convert(page_text).lower().strip())
            
            if _print:
                IO.print_dividing_line("Page {}".format(pn))
                print(p_app_texts[pn-1])
                
        return p_app_texts