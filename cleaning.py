import re
from bs4 import BeautifulSoup
import nltk
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
import spacy

import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_html_tags(text):
    try:
        text = BeautifulSoup(text, "lxml").get_text()
    except Exception:
        # if BeautifulSoup fails for a particular string, leave text as-is
        pass
    return text

def remove_ips(text):
    """Remove IP addresses like 192.168.0.1."""
    return re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '', text)

def remove_full_timestamp(text):
    """Remove full timestamps or date-time combinations like '12:30, 9 October 2007 (UTC)'."""
    return re.sub(r'(?:\d{1,2}:\d{2},?\s*)?(?:\d{1,2}\s\w+,?\s\d{4},?|\w+\s\d{1,2},?\s\d{4},?)(?:\s\([A-Z]{2,4}\))?', '', text)

def remove_timestamp(text):
    """Remove short timestamps like '12:45'."""
    return re.sub(r'\d{1,2}:\d{2}', '', text)

def remove_link(text):
    """Remove URLs starting with http or https."""
    return re.sub(r'https?://\S+', '', text)

def clean_markup_and_quotes(text, start_zone=3):
    """
    This function handles newlines, quotes, stray markup and formatting inconsistencies. It inncludes context-aware newline handling to maintain sentence boundaries.
    """

    # Remove leading commas and extra spaces
    text = re.sub(r'^[,\s]+', '', text)

    # Context-aware newline replacement
    def _newline_repl(match):
        start = match.start()
        if start < start_zone:
            return ''
        prev_char = text[start - 1]
        return ' ' if prev_char in '.!?' else '. '

    text = re.sub(r'\n+', _newline_repl, text)

    # Remove initial stray period and whitespace
    text = re.sub(r'^\s*\.\s*', '', text)

    # Remove various forms of quotation marks and backticks
    text = re.sub(r'[\"\u2018\u2019\u201C\u201D`]', '', text)

    # Remove square brackets and stray hyphens
    text = re.sub(r'[\[\]]+', '', text)
    text = re.sub(r'^\-+\s*', '', text)

    # Remove trailing unmatched parentheses
    text = re.sub(r'\)+\s*$', '', text)

    # Collapse multiple spaces and strip outer whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def clean_text(text):
    """Full cleaning pipeline."""
    text = remove_html_tags(text)
    text = remove_ips(text)
    text = remove_link(text)
    text = remove_full_timestamp(text)
    text = remove_timestamp(text)
    text = clean_markup_and_quotes(text)
    text = remove_accented_chars(text)
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    # expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def lemmatize_text(text):
    # Single string lemmatization for API
    doc = nlp(text)
    return ' '.join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens =  tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_very_long_tokens(text):
    return re.sub(r'\w{30,}|(.{1,5})\1{10,}', '', text)

def clean_text_pipeline(text):
    """
    MASTER FUNCTION FOR API INFERENCE
    Input: Raw String
    Output: Cleaned String
    """
    text = str(text)
    text = clean_text(text)
    text = expand_contractions(text)
    text = lemmatize_text(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    text = remove_very_long_tokens(text)
    
    return text

# --- Batch Functions (To be used ONLY for cleaning before training) ---
def batch_lemmatize(texts):
    """Optimized for processing list of texts using Spacy pipe"""
    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=1000):
        lem = ' '.join([token.lemma_ if token.lemma_ !='-PRON-' else token.text for token in doc])
        lemmatized.append(lem)
    return lemmatized

def preprocess_dataframe(df):
    """
    MASTER FUNCTION FOR TRAINING
    Input: DataFrame
    Output: DataFrame with cleaned column
    """
    # Only import and initialize parallel here, so API doesn't crash
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    print("Cleaning text features...")
    s = df['comment_text']
    s = s.parallel_apply(clean_text)
    s = s.parallel_apply(expand_contractions)
    
    # Batch spacy is faster than parallel apply for this specific task
    print("Lemmatizing...")
    s = pd.Series(batch_lemmatize(s.tolist()), index=s.index)
    s = s.parallel_apply(remove_special_characters)
    s = s.parallel_apply(remove_stopwords)
    s = s.parallel_apply(remove_very_long_tokens)

    df['cleaned_comment_text'] = s
    df = df.replace(r'^(\s?)+$', np.nan, regex=True)
    df = df.dropna().reset_index(drop=True)
    
    return df