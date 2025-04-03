# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:32:37 2025

@author: Admin
"""
#pip install gensim
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

paragraph="""

At the Indian Institute of Technology, Hyderabad, Dr APJ Abdul Kalam gave one of his finest speeches, where he articulated his visions for India.

Here is a transcript of his inspiring speech:

I have three visions for India. In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, The Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture, their history and Tried to enforce our way of life on them. Why? Because we respect the freedom of others.

That is why my first vision is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.

My second vision for India's DEVELOPMENT, For fifty years we have been A developing nation. It is time we see ourselves as a developed nation. We are among top 5 nations of the world in terms of GDP. We have 10 percent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognized today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn't this incorrect?

I have a THIRD vision. India must stand up to the world. Because I believe that, unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of space, Professor Satish Dhawan, who succeeded him and Dr.Brahm Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.

"""
import re

import nltk
nltk.download('punkt')


# Remove content within square brackets
text = re.sub(r'\[[^\]]*\]', '', paragraph)
# Replace multiple spaces with a single space
text = re.sub(r'\s+', ' ', text).strip()
# Convert text to lowercase
text = text.lower()
# Remove digits
text = re.sub(r'\d', '', text)
# Ensure necessary NLTK resources are available
nltk.download('punkt')
# Sentence tokenization
sentences = nltk.sent_tokenize(text)
# Word tokenization
words = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i]=[word for word in sentences[i] if word not in stopwords.words('english')]
    
    
model=Word2Vec(sentences, min_count=2)

words=model.wv.vocab

vector=model.wv['free']

similar=model.wv.most_similar("free")