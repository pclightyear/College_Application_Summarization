from ArticutAPI import Articut
import utils.preprocess as P
import time
import os

username = "# The content is removed due to confidential concerns."
apikey   = "# The content is removed due to confidential concerns."
articut = Articut(
    username,
    apikey,
)

userDefined = "../../var/articut_dict.json"

dummy_reply = {
    'msg': 'Bad request!',
    'status': False
}

def articut_cut(s, lv="lv2", wikiDataBOOL=False, sleep=True):
    if P.is_empty_sent(s):
        return dummy_reply
    
    if sleep:
        time.sleep(0.75)
    
    return articut.parse(
        s, 
        level=lv, 
        wikiDataBOOL=wikiDataBOOL, 
        chemicalBOOL=False,
        userDefinedDictFILE=userDefined
    )

default_unwanted_pos_list = ['ACTION_lightVerb', 'ASPECT', 'AUX', 'MODAL', 'TIME', 'CLAUSE']
def get_tokens(res, remove_punc=True, unwanted_pos_list=['FUNC']):
    result_obj = res['result_obj']
    tokens = []
    
    for sent in result_obj:
        for token in sent:
            ## remove punctuation
            if remove_punc and token['pos'] == "PUNCTUATION":
                continue
                
            ## remove unwanted pos
            no_unwanted_pos = True
            for unwanted_pos in unwanted_pos_list:
                if unwanted_pos in token['pos']:
                    no_unwanted_pos = False
            
            if not no_unwanted_pos:
                continue
            
            tokens.append(token['text'])
                
    return tokens