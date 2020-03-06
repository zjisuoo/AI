import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

t = Tokenizer()
t.fit_on_texts(texts)
print(t.word_index)

# texts is input of 'texts_to_matrix', mode is count
print(t.texts_to_matrix(texts, mode = 'count'))
# texts is input of 'texts_to_matrix', mode is binary
print(t.texts_to_matrix(texts, mode = 'binary'))
# texts is input of 'texts_to_matrix', mode is tfidf, rounding up the 2nd digit
print(t.texts_to_matrix(texts, mode = 'tfidf').round(2))
#  texts is input of 'texts_to_matrix', mode is freq, rounding up the 2nd digit
print(t.texts_to_matrix(texts, mode = 'freq').round(2))