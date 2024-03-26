import re
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET

noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def punctuations_remove(text):

    ar_str = u''.join(UNICODE_PUNCT_SYMBOL_CHARSET) 

    clean_text = re.sub(r'['+re.escape(ar_str)+ r']+', '', text)
    return clean_text

def diacritics_remove(text):
    text = re.sub(noise, '', text)
    return text

def number_remove(text):
    text = re.sub("\d+", "", text)
    return text

def remove_english(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    return text.strip()

def preprocess_senses(text):
    text = punctuations_remove(text)
    text = diacritics_remove(text)
    text = remove_english(text)
    text = number_remove(text)
    text = re.sub('\s+',' ',text)
    return text.strip()