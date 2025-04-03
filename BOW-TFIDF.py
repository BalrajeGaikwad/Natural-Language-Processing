# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

sentences="""Narendra Damodardas Modi[a] (born 17 September 1950)[b] is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a far-right Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.[4]

Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. At the age of 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[d] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[13] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[e] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[f]

In the 2014 Indian general election, Modi led the BJP to a parliamentary majority, the first for a party since 1984. His administration increased direct foreign investment, and reduced spending on healthcare, education, and social-welfare programmes. Modi began a high-profile sanitation campaign, and weakened or abolished environmental and labour laws. His demonetisation of banknotes in 2016 and introduction of the Goods and Services Tax in 2017 sparked controversy. Modi's administration launched the 2019 Balakot airstrike against an alleged terrorist training camp in Pakistan. The airstrike failed,[16][17] but the action had nationalist appeal.[18] Modi's party won the 2019 general election which followed.[19] In its second term, his administration revoked the special status of Jammu and Kashmir,[20][21] and introduced the Citizenship Amendment Act, prompting widespread protests, and spurring the 2020 Delhi riots in which Muslims were brutalised and killed by Hindu mobs.[22][23][24] Three controversial farm laws led to sit-ins by farmers across the country, eventually causing their formal repeal. Modi oversaw India's response to the COVID-19 pandemic, during which, according to the World Health Organization's estimates, 4.7 million Indians died.[25][26] In the 2024 general election, Modi's party lost its majority in the lower house of Parliament and formed a government leading the National Democratic Alliance coalition."""



import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences_list = nltk.sent_tokenize(sentences)
corpus = []

"""for sentence in sentences_list:
    review = re.sub('[^a-zA-Z]', ' ', sentence)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)"""

## Lemmatization
for sentence in sentences_list:
    review = re.sub('[^a-zA-Z]', ' ', sentence)
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

## Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()


## TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(corpus).toarray()

