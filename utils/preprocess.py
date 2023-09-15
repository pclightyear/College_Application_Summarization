import re
from zhon import hanzi

EMPTY_COMMENT = ['0', '0.0', '--', 0]
EMPTY_PATTERN = ['(\d+\.)']
EMPTY_STRING = ' \t\n\r\f\v'
GRADE_COMMENT = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '（A)', '（B)', '（C)', '（D)', '（E)', '（F)', '(A）', '(B）', '(C）', '(D）', '(E）', '(F）', '（A）', '（B）', '（C）', '（D）', '（E）', '（F）']
CH_NUMBER = '一二三四五六七八九十壹貳參肆伍陸柒捌玖拾式'
BULLET_POINT = '★●◆➢'

def is_empty_sent(sent):
    if type(sent) != str:
        return True
    
    if sent == None or sent == '' or sent == 0:
        return True
    ## 0 or 0.0
    for ec in EMPTY_COMMENT:
        if sent == ec:
            return True
    
    for gc in GRADE_COMMENT:
        if sent == gc:
            return True
        
    if sent in EMPTY_STRING:
        return True
        
    return False

def split_sentence(text):
    return list(re.findall(u'[^!?｡。\.\!\?]+[!?｡。\.\!\?]?', text, flags=re.U))

def is_zh_character(ch):
    ## All Chinese characters and punctuations
    re_ch_p = '[{}]'.format(hanzi.characters)
    ## Whitespaces between Chinese characters and punctuations
    return re.findall(re_ch_p, ch) != []

def split_whitespace_btn_ch_character(s):
    ## remove multiple whitespaces
    s = re.sub('\s{2,}', ' ', s)
    
    ## All Chinese characters and punctuations
    re_ch_p = '[{}]'.format(hanzi.characters + hanzi.punctuation)
    ## Whitespaces between Chinese characters and punctuations
    ws_btn_ch = '(?<={})\s(?={})'.format(re_ch_p, re_ch_p)
    
    sent = re.split(ws_btn_ch, s)
    
    return sent

def get_sent_len(s):
    re_alphanumeric = '[a-zA-Z0-9_]+'
    re_ch_p = '[{}]'.format(hanzi.characters + hanzi.punctuation)
    
    l = 0
    
    ## find all english and number token
    l += len(re.findall(re_alphanumeric, s))
    s = re.sub(re_alphanumeric, '', s)
    
    ## remove whitespace
    s = re.sub('\s', '', s)
    
    ## count chinese character
    l += len(re.findall(re_ch_p, s))
    
    return l