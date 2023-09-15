import os
import numpy as np
from zhon import hanzi
import string
from collections import defaultdict

"""
Hyperparameters
"""
TRAIN_PERSPECTIVE_TITLE = {
    # The content is removed due to confidential concerns.
}

ALL_PERSPECTIVE_TITLE = {
    # The content is removed due to confidential concerns.
}

TOP_K = 15
MAX_NUM_PERSPECTIVE = 5

TRAIN_GRADE_LABELS = [['A'], ['B'], ['C'], ['F']]
TRAIN_GRADE_LABEL_EXTENSION = {
    'A': [1, 1, 1],
    'B': [1, 1, 0],
    'C': [1, 0, 0],
    'F': [0, 0, 0],
}
GRADE_LABEL_LOSS_WEIGHT = [15, 3.5, 1, 1.2]
GRADE_ORDINAL_LABEL_LOSS_WEIGHT = [1, 3.5, 15]
RANK_INDEX_TO_LABEL = {
    3: 'A',
    2: 'B',
    1: 'C',
    0: 'F',
}
GRADE_INDEX_TO_LABEL = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'F',
}

"""
Time Info
"""
DATA_BEGIN_YEAR = 106
DATA_END_YEAR = 112

YEAR_LIST = list(range(DATA_BEGIN_YEAR, DATA_END_YEAR + 1))
YEAR_DIRS = [str(year) for year in YEAR_LIST]

GROUP_LABELS = {
    # The content is removed due to confidential concerns.
}



"""
Achievements
"""
MAX_NUM_ACHIEVEMENTS = 5
MAX_NUM_ACHIEVEMENTS_AFTER_YEAR_112 = 3

ACHIEVEMENT_LEVEL_MATCHING = defaultdict(
    # The content is removed due to confidential concerns.
)



"""
Comments
"""
MAX_NUM_COMMITTEE_MEMBER = 6

GRADE_SYMBOLS = ['A', 'B', 'C', 'F']
GRADE_POINTS_MAPPING = {
    'A': 10,
    'B': 8,
    'C': 5,
    'F': 1
}

NULL_GRADE_FILL = {
    'A': 95,
    'B': 85,
    'C': 75,
    'F': 65
}

GRADE_SYMBOLS_NUM_BIN = {
    'A': 6,
    'B': 5,
    'C': 4,
    'D': 3,
    'E': 2,
    'F': 1,
    np.nan: np.nan
}

GRADE_TO_CLASS = {
    'A': 0,
    'B': 1,
    'C': 2,
    'F': 3
}

CLASS_TO_GRADE = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'F'
}



"""
Application
"""
COVER_PAGE_TITLE_LIST = [
    # The content is removed due to confidential concerns.
]

COVER_PAGE_TITLE_DICT = {
    # The content is removed due to confidential concerns.
}



"""
Talent
"""
TALENT_FIELD = [
    # The content is removed due to confidential concerns.
]

TALENT_FIELD_HIERARCHY = {
    # The content is removed due to confidential concerns.
}

TALENT_LIST = [
    # The content is removed due to confidential concerns.
]

TALENT_TO_FIELD_MAPPING = defaultdict(
    # The content is removed due to confidential concerns.
)

"""
Competition Dictionary
"""
COMPETITION_DICT_FIELD_KEYWORD_LIST = [
    # The content is removed due to confidential concerns.
]

COMPETITION_DICT_FIELD_TO_TALENT_FIELD_MAPPING = defaultdict(
    # The content is removed due to confidential concerns.
)

COMPETITION_DICT_TYPE_KEYWORD_LIST = [
    # The content is removed due to confidential concerns.
]
        
COMPETITION_DICT_TYPE_MAPPING = defaultdict(
    # The content is removed due to confidential concerns.
)
    
COMPETITION_DICT_LEVEL_KEYWORD_LIST = [
    # The content is removed due to confidential concerns.
]

COMPETITION_DICT_LEVEL_MAPPING = defaultdict(
    # The content is removed due to confidential concerns.
)

"""
Data Sheet
"""
_106_TO_109_PERSONAL_DATA_SHEET_TITLE = '# The content is removed due to confidential concerns.'
_106_TO_109_PERSONAL_DATA_SHEET_LAST = '# The content is removed due to confidential concerns.'
_106_TO_109_ACHIEVEMENT_DATA_SHEET_TITLE = '# The content is removed due to confidential concerns.'
_106_TO_109_ACHIEVEMENT_DATA_SHEET_LAST = '# The content is removed due to confidential concerns.'

_110_TO_111_DATA_SHEET_LAST = '# The content is removed due to confidential concerns.'



"""
Punctuation & Regex
"""
EN_PUNC_STOPS = "!;?."
EN_PUNC_NON_STOPS = ","
ZH_EN_PUNC_WHITE = hanzi.punctuation + string.punctuation + " \t\n\r\f\v"
FORM_FEED = '\x0c'
NO_DIGIT_SURROUNDING_PERIOD = '(?<=[^\d])(\.)(?=[^\d])'



"""
Visualization
"""
FONT_PATH = os.path.expanduser('~') + '/NotoSansCJKtc/NotoSansMonoCJKtc-Regular.otf'



"""
Articut
"""
ARTICUT_DUMMY_REPLY = {
    'exec_time': 0,
    'result_pos': [''],
    'result_segmentation': '',
    'result_obj': [],
    'level': 'lv2',
    'version': 'v240',
    'status': True,
    'msg': 'Success!',
    'word_count_balance': 0,
    'product': 'https://api.droidtown.co/product/',
    'document': 'https://api.droidtown.co/document/'
}



"""
Others
"""
COMMENT_ARTICUT_WORD_GRAPH_HELP = """
command: python3 1_comment_articut_word_graph.py <option>
options:
    -t <talent field>
    -p <port number>
    -w <word freq filter> 
    -e <edge freq filter>
    -b <span filter>
    -s ['special'|'normal']: special background or normal student only
    -o ['first'|'second']: 1st round applicants or 2nd round applicants only
    -d <word cloud dir>: word cloud's relative directory in /research/results/EDA/qualitative/
"""

COMMENT_SBERT_EMBED_GEN_HELP = """
command: python3 comment_sbert_embed_gen.py <option>
options:
    -g <gpu number>
    -b <batch size>
"""