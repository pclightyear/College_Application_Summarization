import os

YEAR_DIRS = ['106', '107', '108', '109', '110', '111', '112']

"""
./
"""
DATA_DIR = '../../data'
RESULTS_DIR = '../../results'

"""
# ./data
"""
APPLICATIONS_DIR = 'applications'
# ARCHIVE_DIR = 'archive'
COMMENTS_DIR = 'comments'
DICTIONARY_DIR = 'dictionary'
EMBEDS_DIR = 'embeds'
# TOKENIZE_DIR = 'tokenize'

"""
## ./data/applications
"""
DATA_SHEET_DIR = 'data_sheet'
FULL_APPLICATION_DIR = 'full_application'
PROCESSED_DIR = 'processed'
RECOMMENDATION_LETTER_DIR = 'recommendation_letter'

"""
### ./data/applications/data_sheet
"""
CSV_DIR = 'csv'
# XLSX_DIR = 'xlsx'

"""
#### ./data/applications/data_sheet/csv
"""
FP_DATA_SHEET_CSV_DIR = os.path.join(DATA_DIR, APPLICATIONS_DIR, DATA_SHEET_DIR, CSV_DIR)

FP_DATA_SHEET_CSV = []

for year in YEAR_DIRS:  
    fp = os.path.join(FP_DATA_SHEET_CSV_DIR, "{}.csv".format(year))
    FP_DATA_SHEET_CSV.append(fp)

FP_ALL_DATA_SHEET_CSV = os.path.join(FP_DATA_SHEET_CSV_DIR, 'all.csv')
    
"""
### ./data/applications/full_application
"""

"""
#### ./data/applications/full_application/{year}
"""
FP_FULL_APPLICATION_DIR = os.path.join(DATA_DIR, APPLICATIONS_DIR, FULL_APPLICATION_DIR)
PDF_DIR = 'pdf'
TXT_OCR_DIR = 'txt_ocr'
TXT_OCR_RAW_DIR = 'txt_ocr_raw'
IMAGE_DIR = 'image'

FP_FULL_APPLICATIONS_PDF_DIR = []
FP_FULL_APPLICATIONS_TXT_OCR_DIR = []
FP_FULL_APPLICATIONS_TXT_OCR_RAW_DIR = []
FP_FULL_APPLICATIONS_IMAGE_DIR = []

for year in YEAR_DIRS:
    fp = os.path.join(FP_FULL_APPLICATION_DIR, year, PDF_DIR)
    FP_FULL_APPLICATIONS_PDF_DIR.append(fp)
    
    fp = os.path.join(FP_FULL_APPLICATION_DIR, year, TXT_OCR_DIR)
    FP_FULL_APPLICATIONS_TXT_OCR_DIR.append(fp)
    
    fp = os.path.join(FP_FULL_APPLICATION_DIR, year, TXT_OCR_RAW_DIR)
    FP_FULL_APPLICATIONS_TXT_OCR_RAW_DIR.append(fp)
    
    fp = os.path.join(FP_FULL_APPLICATION_DIR, year, IMAGE_DIR)
    FP_FULL_APPLICATIONS_IMAGE_DIR.append(fp)

"""
### ./data/applications/processed/
"""
FP_PROCESSED_APPLICATION_DIR = os.path.join(DATA_DIR, APPLICATIONS_DIR, PROCESSED_DIR)

FP_DF_APPLICANTS_CSV = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_applicants.csv')
FP_DF_APPLICANTS_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_applicants.pkl')

FP_DF_APPLICATIONS_CSV = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_applications.csv')
FP_DF_APPLICATIONS_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_applications.pkl')

FP_DF_ACHIEVEMENTS_CSV = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_achievements.csv')
FP_DF_ACHIEVEMENTS_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_achievements.pkl')

FP_ACHIEVEMENTS_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'achievements.pkl')
FP_ACHIEVEMENTS_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'achievements_sbert_embed.pkl')

FP_DF_RECOMMENDATION_LETTERS_CSV = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_recommendation_letters.csv')
FP_DF_RECOMMENDATION_LETTERS_PKL = os.path.join(FP_PROCESSED_APPLICATION_DIR, 'df_recommendation_letters.pkl')

"""
### ./data/applications/recommendation_letter
"""
CSV_DIR = 'csv'
# XLSX_DIR = 'xlsx'

"""
#### ./data/applications/recommendation_letter/csv
"""
FP_RECOMMENDATION_LETTER_DIR_CSV_DIR = os.path.join(DATA_DIR, APPLICATIONS_DIR, RECOMMENDATION_LETTER_DIR, CSV_DIR)

FP_RECOMMENDATION_LETTER_CSV = []

for year in YEAR_DIRS:  
    fp = os.path.join(FP_RECOMMENDATION_LETTER_DIR_CSV_DIR, "{}.csv".format(year))
    FP_RECOMMENDATION_LETTER_CSV.append(fp)

FP_ALL_RECOMMENDATION_LETTER_CSV = os.path.join(FP_RECOMMENDATION_LETTER_DIR_CSV_DIR, 'all.csv')

    
    
"""
## ./data/comments
"""
# RAW_DIR = 'raw'
CSV_DIR = 'csv'
# XLSX_DIR = 'xlsx'
PROCESSED_DIR = 'processed'

"""
### ./data/comments/csv
"""
FP_COMMENT_CSV_DIR = os.path.join(DATA_DIR, COMMENTS_DIR, CSV_DIR)

FP_COMMENT_CSV = []

for year in YEAR_DIRS:  
    fp = os.path.join(FP_COMMENT_CSV_DIR, "{}.csv".format(year))
    FP_COMMENT_CSV.append(fp)

FP_ALL_COMMENT_CSV = os.path.join(FP_COMMENT_CSV_DIR, 'all.csv')
    
"""
### ./data/comments/processed
"""
FP_PROCESSED_COMMENT_DIR = os.path.join(DATA_DIR, COMMENTS_DIR, PROCESSED_DIR)

FP_DF_COMMENTS_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_comments.csv')
FP_DF_COMMENTS_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_comments.pkl')

FP_DF_SPLIT_COMMENTS_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments.csv')
FP_DF_SPLIT_COMMENTS_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments.pkl')
FP_SPLIT_COMMENTS_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments.pkl')
FP_SPLIT_COMMENTS_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_sbert_embed.pkl')

FP_DF_SPLIT_COMMENTS_NO_DUPLICATE_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_no_duplicate.csv')
FP_DF_SPLIT_COMMENTS_NO_DUPLICATE_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_no_duplicate.pkl')
FP_SPLIT_COMMENTS_NO_DUPLICATE_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_no_duplicate.pkl')
FP_SPLIT_COMMENTS_NO_DUPLICATE_ARTICUT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_no_duplicate_articut.pkl')
FP_SPLIT_COMMENTS_NO_DUPLICATE_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_no_duplicate_sbert_embed.pkl')

FP_DF_SPLIT_COMMENTS_NSP_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_nsp.csv')
FP_DF_SPLIT_COMMENTS_NSP_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_nsp.pkl')
FP_SPLIT_COMMENTS_NSP_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_nsp.pkl')
FP_SPLIT_COMMENTS_NSP_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_nsp_sbert_embed.pkl')

FP_DF_SPLIT_COMMENTS_NSP_NO_DUPLICATE_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_nsp_no_duplicate.csv')
FP_DF_SPLIT_COMMENTS_NSP_NO_DUPLICATE_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_split_comments_nsp_no_duplicate.pkl')
FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_nsp_no_duplicate.pkl')
FP_SPLIT_COMMENTS_NSP_NO_DUPLICATE_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_nsp_no_duplicate_sbert_embed.pkl')

FP_SPLIT_COMMENTS_TRAIN_TEST_INDICES = os.path.join(FP_PROCESSED_COMMENT_DIR, 'split_comments_train_test_indices.pkl')

FP_DF_TOKENIZED_COMMENTS_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_tokenized_comments.csv')
FP_DF_TOKENIZED_COMMENTS_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'df_tokenized_comments.pkl')

FP_ENG_COMMENT_LABEL_CSV = os.path.join(FP_PROCESSED_COMMENT_DIR, 'eng_comment_label.csv')

FP_TRAIN_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'train_comment_sentiment.pkl')
FP_TEST_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'test_comment_sentiment.pkl')
FP_ALL_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'all_comment_sentiment.pkl')

FP_TRAIN_AUG_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'train_aug_comment_sentiment.pkl')
FP_TEST_AUG_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'test_aug_comment_sentiment.pkl')
FP_ALL_AUG_COMMENT_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'all_aug_comment_sentiment.pkl')

FP_TRAIN_SPLIT_COMMENTS_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'train_split_comments_sentiment.pkl')
FP_ALL_SPLIT_COMMENTS_SENTIMENT_PKL = os.path.join(FP_PROCESSED_COMMENT_DIR, 'all_split_comments_sentiment.pkl')

"""
## ./data/dictionary
"""
RAW_DIR = 'raw'
CSV_DIR = 'csv'
PROCESSED_DIR = 'processed'

"""
### ./data/dictionary/csv
"""
FP_DICTIONARY_CSV_DIR = os.path.join(DATA_DIR, DICTIONARY_DIR, CSV_DIR)

FP_COMPETITION_DICT_CSV = os.path.join(FP_DICTIONARY_CSV_DIR, 'competition_dictionary_1101005.csv')

"""
### ./data/dictionary/processed
"""
FP_PROCESSED_DICTIONARY_DIR = os.path.join(DATA_DIR, DICTIONARY_DIR, PROCESSED_DIR)

FP_DF_COMPETITION_DICT_CSV = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'df_competition_dict.csv')
FP_DF_COMPETITION_DICT_PKL = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'df_competition_dict.pkl')

FP_DF_COMPETITION_DICT_CSV = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'df_competition_dict.csv')
FP_DF_COMPETITION_DICT_PKL = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'df_competition_dict.pkl')

FP_COMPETITION_DICT_ITEMS_PKL = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'competition_dict_items.pkl')
FP_COMPETITION_DICT_ITEMS_SBERT_EMBED_PKL = os.path.join(FP_PROCESSED_DICTIONARY_DIR, 'competition_dict_items_sbert_embed.pkl')


"""
## ./data/embeds
"""
CKIP_DIR = 'CKIP'
FASTTEST_DIR = 'fasttext'
UIO_DIR = 'UiO'

"""
### ./data/embeds/CKIP
"""
FP_CKIP_EMBEDS_DIR = os.path.join(DATA_DIR, EMBEDS_DIR, CKIP_DIR)

FP_CKIP_W2V = os.path.join(FP_CKIP_EMBEDS_DIR, 'w2v_CNA_ASBC_300d.vec')
FP_CKIP_GLOVE = os.path.join(FP_CKIP_EMBEDS_DIR, 'Glove_CNA_ASBC_300d.vec')

"""
### ./data/embeds/fasttext
"""
FP_CKIP_EMBEDS_DIR = os.path.join(DATA_DIR, EMBEDS_DIR, CKIP_DIR)

FP_FASTTEXT_BIN = os.path.join(FP_CKIP_EMBEDS_DIR, 'cc.zh.300.bin')

"""
### ./data/embeds/UiO
"""
FP_CKIP_EMBEDS_DIR = os.path.join(DATA_DIR, EMBEDS_DIR, CKIP_DIR)

FP_UIO_BIN = os.path.join(FP_CKIP_EMBEDS_DIR, 'model.bin')
FP_UIO_TXT = os.path.join(FP_CKIP_EMBEDS_DIR, 'model.txt')



"""
# ./results
"""
CKIP_COMMENT_POS_NER_CLUSTERING_DIR = 'ckip_comment_pos_ner_clustering'
EDA_DIR = 'EDA'
LDA_DIR = 'LDA'
SIGNIFICANCE_PHAN_DIR = 'significance_pHAN'
UNIQUENESS_PHAN_DIR = 'uniqueness_pHAN'
SIGNIFICANCE_PSEUDO_SUMMARY_DIR = 'significance_pseudo_summary'
UNIQUENESS_PSEUDO_SUMMARY_DIR = 'uniqueness_pseudo_summary'
SENTENCE_CLUSTERING_DIR = 'sentence_clustering'
SPAN_DETECTION_DIR = 'span_detection'
SUMMARY_DIR = 'summary'

FP_CKIP_COMMENT_POS_NER_CLUSTERING_DIR = os.path.join(RESULTS_DIR, CKIP_COMMENT_POS_NER_CLUSTERING_DIR)
FP_EDA_DIR = os.path.join(RESULTS_DIR, EDA_DIR)
FP_LDA_DIR = os.path.join(RESULTS_DIR, LDA_DIR)
FP_SIGNIFICANCE_PHAN_DIR = os.path.join(RESULTS_DIR, SIGNIFICANCE_PHAN_DIR)
FP_UNIQUENESS_PHAN_DIR = os.path.join(RESULTS_DIR, UNIQUENESS_PHAN_DIR)
FP_SIGNIFICANCE_PSEUDO_SUMMARY_DIR = os.path.join(RESULTS_DIR, SIGNIFICANCE_PSEUDO_SUMMARY_DIR)
FP_UNIQUENESS_PSEUDO_SUMMARY_DIR = os.path.join(RESULTS_DIR, UNIQUENESS_PSEUDO_SUMMARY_DIR)
FP_SENTENCE_CLUSTERING_DIR = os.path.join(RESULTS_DIR, SENTENCE_CLUSTERING_DIR)
FP_SPAN_DETECTION_DIR = os.path.join(RESULTS_DIR, SPAN_DETECTION_DIR)
FP_SUMMARY_DIR = os.path.join(RESULTS_DIR, SUMMARY_DIR)

"""
## ./results/EDA
"""
QUALITATIVE_DIR = 'qualitative'
QUANTITATIVE_DIR = 'quantitative'

FP_EDA_QUALITATIVE_DIR = os.path.join(FP_EDA_DIR, QUALITATIVE_DIR)
FP_EDA_QUANTITATIVE_DIR = os.path.join(FP_EDA_DIR, QUANTITATIVE_DIR)

"""
## ./results/LDA
"""

"""
## ./results/pHAN
"""

"""
## ./results/significance_pseudo_summary
"""

"""
## ./results/uniqueness_pseudo_summary
"""

"""
## ./results/sentence_clustering
"""
COMMENTS_DIR = 'comments'

"""
### ./results/sentence_clustering/comments
"""
MODEL_DIR = 'model'
PLOT_DIR = 'plot'
TOPIC_CATEGORY_MAPPING_DIR = 'topic_category_mapping'
TOPIC_HIERARCHY_DIR = 'topic_hierarchy'

"""
#### ./results/sentence_clustering/comments/model
"""
FP_COMMENT_CLUSTERING_MODEL_DIR = os.path.join(FP_SENTENCE_CLUSTERING_DIR, COMMENTS_DIR, MODEL_DIR)

FP_COMMENT_CLUSTERING_HDBSCAN_PACKAGE = os.path.join(FP_COMMENT_CLUSTERING_MODEL_DIR, 'comment_clustering_hdbscan_package.pkl')
FP_COMMENT_CLUSTERING_BISECTING_KMEANS_PACKAGE = os.path.join(FP_COMMENT_CLUSTERING_MODEL_DIR, 'comment_clustering_bisecting_kmeans_package.pkl')

"""
#### ./results/sentence_clustering/comments/plot
"""
FP_COMMENT_CLUSTERING_PLOT_DIR = os.path.join(FP_SENTENCE_CLUSTERING_DIR, COMMENTS_DIR, PLOT_DIR)

HDBSCAN_DIR = 'HDBSCAN'
BISECTING_KMEANS_DIR = 'Bisecting_KMeans'

FP_COMMENT_CLUSTERING_PLOT_HDBSCAN_DIR = os.path.join(FP_COMMENT_CLUSTERING_PLOT_DIR, HDBSCAN_DIR)
FP_COMMENT_CLUSTERING_PLOT_BISECTING_KMEANS_DIR = os.path.join(FP_COMMENT_CLUSTERING_PLOT_DIR, BISECTING_KMEANS_DIR)

"""
#### ./results/sentence_clustering/comments/topic_category_mapping
"""
FP_COMMENT_CLUSTERING_TOPIC_CATEGORY_MAPPING_DIR = os.path.join(FP_SENTENCE_CLUSTERING_DIR, COMMENTS_DIR, TOPIC_CATEGORY_MAPPING_DIR)

"""
#### ./results/sentence_clustering/comments/topic_hierarchy
"""
FP_COMMENT_CLUSTERING_TOPIC_HIERARCHY_DIR = os.path.join(FP_SENTENCE_CLUSTERING_DIR, COMMENTS_DIR, TOPIC_HIERARCHY_DIR)

"""
## ./results/span_detection
"""

"""
## ./results/summary
"""
GENERATION_DIR = 'generation'
EVALUATION_DIR = 'evaluation'
SIGNIFICANCE_SUMMARY = 'significance_summary'
UNIQUENESS_SUMMARY = 'uniqueness_summary'

FP_SUMMARY_GENERATION_DIR = os.path.join(FP_SUMMARY_DIR, GENERATION_DIR)
FP_SUMMARY_EVALUATION_DIR = os.path.join(FP_SUMMARY_DIR, GENERATION_DIR)
FP_SIGNIFICANCE_SUMMARY_DIR = os.path.join(FP_SUMMARY_DIR, SIGNIFICANCE_SUMMARY)
FP_UNIQUENESS_SUMMARY_DIR = os.path.join(FP_SUMMARY_DIR, UNIQUENESS_SUMMARY)

"""
### ./results/summary/generation
"""
DOCX_DIR = 'docx'

FP_DF_SUMMARY_CSV = os.path.join(FP_SUMMARY_GENERATION_DIR, 'df_summary.csv')
FP_DF_SUMMARY_PKL = os.path.join(FP_SUMMARY_GENERATION_DIR, 'df_summary.pkl')

FP_SUMMARY_DOCX_DIR = os.path.join(FP_SUMMARY_GENERATION_DIR, DOCX_DIR)

"""
### ./results/summary/evaluation
"""
BERTSCORE_DIR = 'BERTScore'
ROUGE_DIR = 'ROUGE'

FP_BERTSCORE_DIR = os.path.join(FP_SUMMARY_EVALUATION_DIR, BERTSCORE_DIR)
FP_ROUGE_DIR = os.path.join(FP_SUMMARY_EVALUATION_DIR, ROUGE_DIR)

"""
#### ./results/summary/evaluation/BERTScore
"""
SAMPLE_PLOT_DIR = 'sample_plot'

FP_BERTSCORE_SAMPLE_PLOT_DIR = os.path.join(FP_BERTSCORE_DIR, SAMPLE_PLOT_DIR)
