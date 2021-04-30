import numpy as np
import math
from utils import spam_words_list

print(spam_words_list)

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug

    def _compute_misspelled_words(self, email):
        """
        Computes the number of misspelled words in sentence
        """
        from nltk.corpus import words
        
        num_misspelled = 0
        word_list = email.split()
        for word in word_list:
            if word in words.words():
                num_misspelled += 1
        return (num_misspelled)
    
    def _compute_spam_phrases_count(self, email):
        """
        Computes the number of known spam words in sentence
        """
        from nltk.corpus import words
        
        num_spam = 0
        word_list = email.split()
        for word in spam_words_list:
            if word in words.words():
                num_spam += 1
        return (num_spam)
    
    def _compute_exclamation_point_count(self, email):
        """
        Computes the number of exclamation points in sentence
        """
        from nltk.corpus import words
        
        num_char = 0
        for character in email:
            num_char += 1
        return (num_char)
    
    def extract_features(self, email, debug=True):
        """
        Here is where you will extract your features from the data in 
        the given window.
        
        Make sure that x is a vector of length d matrix, where d is the number of features.
        
        """        
        x = []
        
        x = np.append(x, self._compute_misspelled_words(email))
        x = np.append(x, self._compute_spam_phrases_count(email))
        x = np.append(x, self._compute_exclamation_point_count(email))

        # convert the list of features to a single 1-dimensional vector
        feature_vector = np.array(x, dtype="object").flatten()
        return feature_vector
