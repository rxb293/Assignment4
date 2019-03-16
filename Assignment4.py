#!/usr/bin/env python
# coding: utf-8

# In[267]:


import os
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from IPython.core.display import HTML
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import spacy
import copy
import elasticsearch

nlp = spacy.load('en_core_web_sm')


# In[474]:


# Preprocessing 
def extractWHword(text):
    match = re.search('(?:Who|What|When|Where|Why|Which|Whom|Whose)',text)
    return match.group()

def stopwordremoval(text):
    return [word for word in re.split('\s',text) if word.lower() not in stopwords.words('english')]

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in re.split('\s',text)]

def stemmer(text):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(token) for token in word_tokenize(text)]

def removepunc(s):
    return re.sub(r'[^\w\s%]','',s)

def joinlist(listoftext):
    return ' '.join(listoftext)

def tokenindex(text, token):
    tokenized = word_tokenize(text)
    for i in range(len(tokenized)):
        if tokenized[i] == token:
            return i
        
#Define grab word functions
def grab_word_before(text,name):
    sentence = word_tokenize(text)
    indexfirstname = tokenindex(text,word_tokenize(name)[0])
    indexlastname = tokenindex(text,word_tokenize(name)[len(word_tokenize(name))-1])
    if indexlastname == len(word_tokenize(text)) - 1 and indexfirstname!= None: #If last name is in end of sentence
        return sentence[indexfirstname - 1]
    elif(indexfirstname != None and indexlastname!= None): #If name is in middle of sentence
        return sentence[indexfirstname - 1]
    else:
        return 'NULL'
def grab_word_after(text,name):
    sentence = word_tokenize(text)
    indexfirstname = tokenindex(text,word_tokenize(name)[0])
    indexlastname = tokenindex(text,word_tokenize(name)[len(word_tokenize(name))-1])
    if indexlastname == (len(sentence) - 1) and indexfirstname != None: #If last name is in end of sentence
        return 'NULL'
    elif indexfirstname == 0 and indexlastname != None and indexlastname != 1: #if first name is in beginning of sentence and there are more to the sentence
        return sentence[indexlastname+1]
    elif(indexfirstname != None and indexlastname!= None): # If name is in middle of sentence
        return sentence[indexlastname+1]
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
def hasceoindicator(inputString):
    ceoindicators = ['ceo','chair','chairman','chairwoman','executive','investor','founder','chief']
    if inputString.lower() in ceoindicators:
        return True
    else:
        return False
def havebusinesswords(inputString):
    businesswordindicators = ['yoy','growth','strategy','stock','profit','loss','company','Corporation']
    words_re = re.compile("|".join(businesswordindicators))
    if words_re.search(inputString.lower()):
        return True
    else:
        return False
def sentencehasceoindicator(inputString):
    ceoindicators = ['ceo','executive','investor','founder']
    words_re = re.compile("|".join(ceoindicators))
    if words_re.search(inputString.lower()):
        return True
    else:
        return False
def hascompanyindicator(inputString):
    companyindicators = ['Inc','Corp','Corporation','Bank','LLC','Group','Ltd','Ventures','Capital','Partners','Company','Holdings']
    words_re = re.compile("|".join(companyindicators))
    if words_re.search(inputString)!=None:
        return True
    else:
        return False
    
#### Define function that find names from the entire corpus

import nltk
from nameparser.parser import HumanName

def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    return (person_list)

def senttobigram(sent):
    return list(nltk.bigrams(stopwordremoval(removepunc(sent))))
def bigrammatches(keyword_bigram,sentence_bigram):
    matches = 0
    for i in range(len(keyword_bigram)):
        for j in range(len(sentence_bigram)):
            if keyword_bigram[i] == sentence_bigram[j]:
                matches += 1
    return matches

def findorg(text):
    doc = nlp(text)
    ents = [(e.text, e.label_) for e in doc.ents]
    orgs = []
    for i in range(len(ents)):
        if ents[i][1] == 'ORG':
            orgs.append(ents[i][0])
    return orgs


# # Argument parsing

# In[ ]:


parser = argparse.ArgumentParser(description="Returns answer(s) to question")


# In[ ]:


parser.add_argument("question",help="The question itself",type=str)
args = parser.parse_args()


# In[ ]:


question = args.question


# # Question Analysis


# In[1333]:


# Extract WH_word
WH_word = extractWHword(question)


# In[1334]:


# Extract Basic Keywords after removing Stopwords & Punctuation, and Lemmatizing the text
def extractbasickeywords(text):
    kw = []
    tokenize = lemmatization(removepunc(joinlist(stopwordremoval(text))))
    pos = nltk.pos_tag(tokenize)
    for i in range(len(pos)):
        postag = pos[i][1]
        word = pos[i][0]
        if postag in ['NN','NNP','JJ','CD']:
            kw.append(word)
    return kw


# In[1335]:


basic_kw = extractbasickeywords(question)


# In[1336]:


# Similar month abbreviations
def monthabb(month):
    if month == 'January':
        return ['01','Jan']
    if month == 'February':
        return ['02','Feb']
    if month == 'March':
        return ['03','Mar']
    if month == 'April':
        return ['04','Apr']
    if month == 'May':
        return ['05']
    if month == 'June':
        return ['06','Jun']
    if month == 'July':
        return ['07','Jul']
    if month == 'August':
        return ['08','Aug']
    if month == 'September':
        return ['09','Sep']
    if month == 'October':
        return ['10','Oct']
    if month == 'November':
        return ['11','Nov']
    if month == 'December':
        return ['12','Dec']


# In[1344]:


# Extend Keywords depending on question type:
# For question type: 'Which companies went bankrupt in month February of year 2015?'
kw = copy.deepcopy(basic_kw)
if WH_word == 'Which':
    # Extend Month Abbreviations
    monthindex = tokenindex(joinlist(kw),"month")+1
    yearindex = tokenindex(joinlist(kw),"year")+1
    kw.extend(monthabb(kw[monthindex]))
    # Extend company indicators
    kw.extend(['Inc','Corp','Corporation','Bank','LLC','Group','Ltd','Ventures','Capital','Partners','Company','Holdings','bankruptcy','declared','filed'])
# For question type: 'What affects GDP?' and ''What percentage of drop or increase in GDP is associated with unemployment?''
if WH_word == 'What':
    factor_keyword = basic_kw[len(basic_kw)-1]
    kw.extend(['increase','decrease','drop','rise','factor','influence','associated','lead','grow','shrink','%','percent'])

# For question type: 'Who is the CEO of Apple Inc? ==> No extension

#Remove Duplicates
kw = list(dict.fromkeys(kw))


# # Import Index

# In[395]:


import whoosh.index as index

ix = index.open_dir("indexdir")


# # Subset to documents with top 10 highest Okapi scores

# - factory enables search to score documents that contain more of the words higher

# In[1346]:


from whoosh import qparser
# Factory: let documents that contain more of the words searched for to score higher
og = qparser.OrGroup.factory(0.9)
parser = qparser.QueryParser("content", ix.schema, group=og)
myquery = parser.parse(joinlist(kw))

# Search for documents that contain the keywords and select top 5 based on the default OKAPI scoring mechanism
searcher = ix.searcher()

if WH_word == 'Who':
    querylimit = 5
    results = searcher.search(myquery,limit = querylimit,terms=True)
    print("Number of Documents Found:",len(results))
elif WH_word =="Which":
    querylimit = 10
    results = searcher.search(myquery, limit = querylimit,terms=True)
    print("Number of Documents Found:",len(results))
elif WH_word =="What":
    querylimit = 200
    results = searcher.search(myquery, limit = querylimit,terms=True)
    print("Number of Documents Found:",len(results))

print("Number of Documents Selected:",querylimit)
# Print score
#for hit in results:
#    print("Rank", hit.rank)
#    print("Score", hit.score)
#    print("Document Number", hit.docnum) 

    
# Convert Query to sentences
articles = []
i = 0
for i in range(querylimit):
    articles.append(results[i]['content'])
    
# Sent_tokenize: Break Text into sentences
sentences = [sent_tokenize(article) for article in articles]
# Tokenize sentences
tokenized_sentences = []
complete_sentences = []
count = 0
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        tokenized_sentences.append(re.split('\s',sentences[i][j]))
        complete_sentences.append(sentences[i][j])
df_CS = pd.DataFrame(complete_sentences,columns=['text'])


# # Employ Classifiers to trim down dataset

# In[1351]:


if WH_word == 'What':

    #Subset to only sentences containing percentages
    def findpercentage(text):
        pat = re.compile(r'([\d\w\-.])+(\%|\s\%|\s\b[Pp]ercent\b|\s\b[Pp]ercentage\spoint\b|\s\b[Pp]ercentage\spoints\b|\s\b[Pp]ercentage\b|\s\b[Pp]ercentile\spoint\b|\s\b[Pp]ercentile\spoints\b)')
        listofpercentage = []
        if re.finditer !=None: 
            for match in re.finditer(pat,text):
                listofpercentage.append(match.group(0))
        return listofpercentage

    # tokenize text - remember to convert text to lower case
    df_CS['percentages'] = df_CS['text'].apply(lambda x: findpercentage(x))
    # Extract sentence-percentage pair
    s2 = df_CS[df_CS['percentages'].map(lambda d: len(d)) > 0]
    s2.reset_index(drop = True,inplace =True)
    res2 = s2.set_index(['text'])['percentages'].apply(pd.Series).stack()
    res2 = res2.reset_index()
    res2.drop(['level_1'],axis = 1,inplace = True)
    res2.columns = ['text','percentages']
    df_CS = res2
    #Drop Sentences that were duplicated because of multiple percentages
    df_CS.drop_duplicates(subset="text",keep=False,inplace=True)
    
    if question == "What affects GDP?":
        #Subset to only sentences that contains "gdp"
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        response = vectorizer.fit_transform(df_CS.text)
        
        ###### Grab index of keywords in feature list, compute sum of tfidf of keywords for each sentence 
        kwindex = []
        gdp_GDP = ['gdp']
        for i in range(len(gdp_GDP)):
            if gdp_GDP[i].lower() in vectorizer.get_feature_names():
                kwindex.append(vectorizer.get_feature_names().index(gdp_GDP[i].lower()))

        ##Compute sum of tf-idf score for kw for each sentence
        score = [0] * (response.shape[0])
        for i in range(response.shape[0]):
            for j in range(len(kwindex)):
                score[i] += response[i,kwindex[j]]

        df_CS['score'] = score

        df_CS.sort_values(by = 'score',ascending=False)

        df_CS = df_CS.sort_values(by = 'score',ascending=False)[df_CS['score'] != 0]

        # Extract only nouns in sentences
        df_CS['text'] = df_CS['text'].apply(lambda x: nltk.pos_tag(word_tokenize(x)))

        def extract_NN(pos_list):
            returnlist = []
            for x in pos_list:
                if x[1] in ['NN','NNP']:
                    returnlist.append(x[0])
            return returnlist
        df_CS['text'] = df_CS['text'].apply(lambda x: joinlist(extract_NN(x)))
        # Only capture sentences longer than 10 words. Shorter sentences tend to have less meaning.
        df_CS['sent_len'] = df_CS['text'].apply(lambda x: len(word_tokenize(x)))
        df_CS_2 = df_CS[df_CS['sent_len']>10]
        #sns.scatterplot(x=df_CS_2['sent_len'],y=df_CS_2['score'])

        response2 = vectorizer.fit_transform(df_CS_2.text)
        words = vectorizer.get_feature_names()
        word_score = response2.sum(axis=0).tolist()[0]
        df_words = pd.DataFrame(data = words,columns = ['words'])
        df_words['score'] = word_score
        df_words.sort_values(by = 'score',ascending = False,inplace=True)
        df_words['word_len'] = df_words['words'].apply(lambda x: len(x))
        # Only longer nouns tend to be factors
        df_final = df_words[df_words['word_len']>9]
        #sns.scatterplot(x=df_words['word_len'],y=df_words['score'])
        #format df_final consistent with other tables
        df_final.columns = ['names','score','word_len']
    else:
        df_final = df_CS.copy(deep=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        response = vectorizer.fit_transform(df_final.text)
        
        # Grab bigrams of keywords in question
        keyword_bigram = list(nltk.bigrams(basic_kw))
        # convert setences into bigrams
        df_final['sentence_bigrams'] = df_final['text'].apply(lambda x:senttobigram(x))
        df_final['bigrammatchscore'] = df_final['sentence_bigrams'].apply(lambda x: bigrammatches(keyword_bigram,x))

        # Grab index of keywords in feature list, compute sum of tfidf of keywords for each sentence 
        kwindex = []
        for i in range(len(kw)):
            if kw[i].lower() in vectorizer.get_feature_names():
                kwindex.append(vectorizer.get_feature_names().index(kw[i].lower()))

        # Grab index of features that are not keywords
        allindex = list(range(len(vectorizer.get_feature_names())))
        notkwindex = [x for x in allindex if x not in basic_kw] #Only use basickw here so that extended list won't have large weight

        # Grab "GDP index for each sentence
        GDPkeywords = ['gdp','GDP',factor_keyword]
        GDPkeywordindex = []
        
        for i in range(len(GDPkeywords)):
            if GDPkeywords[i].lower() in vectorizer.get_feature_names():
                GDPkeywordindex.append(vectorizer.get_feature_names().index(GDPkeywords[i].lower()))
        
        # Grab factorword index for each sentence
        factorkeywords = [factor_keyword]
        factorkeywordindex = []
        score_GDP = [0] * (response.shape[0])
        for i in range(len(factorkeywords)):
            if factorkeywords[i].lower() in vectorizer.get_feature_names():
                factorkeywordindex.append(vectorizer.get_feature_names().index(factorkeywords[i].lower()))
                
        #Compute sum of tf-idf score for kw for each sentence
        score_kw = [0] * (response.shape[0])
        score_not_kw = [0] * (response.shape[0])
        score_GDP = [0] * (response.shape[0])
        score_factor = [0] * (response.shape[0])
        for i in range(response.shape[0]):
            # Score of keywords
            score_kw[i] = response[i,kwindex].sum(axis=1).tolist()[0][0]
            # Score of non-keywords
            score_not_kw[i] = response[i,notkwindex].sum(axis=1).tolist()[0][0]
            # Score of GDP
            score_GDP[i] = response[i,GDPkeywordindex].sum(axis=1).tolist()[0][0]
            # Score of factors
            score_factor[i] = response[i,factorkeywordindex].sum(axis=1).tolist()[0][0]
        df_final['score'] = list(map(operator.add, list(df_final['bigrammatchscore']), score_kw))
        #Not subtracting score of non-keywords because it doesn't work well.
        #df_final['score'] = list(map(operator.sub,list(df_final['score']),score_not_kw))
        df_final['score_GDP'] = score_GDP
        df_final['score_factor'] = score_factor
        df_final = df_final[df_final['score_GDP']!=0].sort_values(by = 'score',ascending=False)
        df_final = df_final[df_final['score_factor']!=0].sort_values(by = 'score',ascending=False)
        
        
        
        


# In[1155]:


if WH_word == 'Who':
    df_CS['names'] = df_CS['text'].apply(lambda x: get_human_names(x))
    s = df_CS[df_CS['names'].map(lambda d: len(d)) > 0]
    s.reset_index(drop = True,inplace =True)
    res = s.set_index(['text'])['names'].apply(pd.Series).stack()
    res = res.reset_index()
    res.drop(['level_1'],axis = 1,inplace = True)
    res.columns = ['text','names']
    res['RemovedStopwords'] = res['text'].apply(stopwordremoval).apply(joinlist)
    res['Lemmatized'] = res['RemovedStopwords'].apply(lemmatization).apply(joinlist)
    res['Lemmatized'] = res['Lemmatized'].apply(lambda x: removepunc(x))
    res['names'] = res['names'].apply(lambda x: removepunc(x))
    res['word_before'] = res.apply(lambda x : grab_word_before(x['Lemmatized'],x['names']),axis=1)
    res['word_after'] = res.apply(lambda x : grab_word_after(x['Lemmatized'],x['names']),axis=1)
    df_CS_feed = res.copy(deep=True)
    df_CS_feed = df_CS_feed[df_CS_feed['word_after'].isnull() == False]
    df_CS_feed.reset_index(drop = True,inplace = True)
    # Extract Name Features
    df_CS_feed['Capitalized'] = False
    df_CS_feed['lengthofname'] = 0
    df_CS_feed['lengthoftoken'] = 0
    df_CS_feed['nameinbeg'] = False
    df_CS_feed['nameinend'] = False

    df_CS_feed['Capitalized'] = df_CS_feed['names'].apply(lambda x: x.istitle())
    df_CS_feed['lengthofname'] = df_CS_feed['names'].apply(lambda x: len(x))
    df_CS_feed['lengthoftoken'] = df_CS_feed['names'].apply(lambda x: len(word_tokenize(x)))
    df_CS_feed['nameinbeg'] = df_CS_feed.apply(lambda x: tokenindex(x['Lemmatized'], word_tokenize(x['names'])[0])==0,axis = 1)
    df_CS_feed['nameinend'] = df_CS_feed.apply(lambda x: tokenindex(x['Lemmatized'], word_tokenize(x['names'])[len(word_tokenize(x['names']))-1]) == len(word_tokenize(x['Lemmatized']))-1,axis = 1)

    # Extract Before and After Word Features
    df_CS_feed['beforewordlength'] = 0
    df_CS_feed['beforewordcapitalized'] = False
    df_CS_feed['beforewordcontainnumbers'] = False
    df_CS_feed['beforewordcontainceoindicator'] = False
    df_CS_feed['afterwordlength'] = 0
    df_CS_feed['afterwordcapitalized'] = False
    df_CS_feed['afterwordcontainnumbers'] =False
    df_CS_feed['afterwordcontainceoindicator'] = False
    df_CS_feed['beforewordlength'] = df_CS_feed['word_before'].apply(lambda x: len(x))
    df_CS_feed['beforewordcapitalized'] = df_CS_feed['word_before'].apply(lambda x: x.istitle())
    df_CS_feed['beforewordcontainnumbers'] = df_CS_feed['word_before'].apply(lambda x: hasNumbers(x))
    df_CS_feed['beforewordcontainceoindicator'] = df_CS_feed['word_before'].apply(lambda x: hasceoindicator(x))
    df_CS_feed['afterwordlength'] = df_CS_feed['word_after'].apply(lambda x: len(x))
    df_CS_feed['afterwordcapitalized'] = df_CS_feed['word_after'].apply(lambda x: x.istitle())
    df_CS_feed['afterwordcontainnumbers'] =df_CS_feed['word_after'].apply(lambda x: hasNumbers(x))
    df_CS_feed['afterwordcontainceoindicator'] = df_CS_feed['word_after'].apply(lambda x: hasceoindicator(x))

    # Extract Sentence Feature
    df_CS_feed['senetencecontainceo'] = df_CS_feed['Lemmatized'].apply(lambda x: sentencehasceoindicator(x))
    df_CS_feed['senetencecontainbusinesswords'] = df_CS_feed['Lemmatized'].apply(lambda x: havebusinesswords(x))
    
    # Stem words before and after to feed in getdummy
    df_CS_feed['word_before'] = df_CS_feed['word_before'].apply(stemmer).apply(joinlist)
    df_CS_feed['word_after'] = df_CS_feed['word_after'].apply(stemmer).apply(joinlist)

    # Load classifiers and onehotencoder
    from sklearn.preprocessing import OneHotEncoder
    from joblib import dump, load
    clf = joblib.load('ceotrainer.joblib')
    encbefore = joblib.load('ceobeforeencode.joblib')
    encafter = joblib.load('ceoafterencode.joblib')
    
    beforeencode = encbefore.transform(df_CS_feed.word_before.values.reshape(-1,1)).toarray()
    dfOneHotBefore = pd.DataFrame(beforeencode,columns = ['before_'+str(int(i)) for i in range (beforeencode.shape[1])])
    dfOneHotBefore.reset_index(drop=True,inplace=True)
    df_CS_feed = pd.concat([df_CS_feed,dfOneHotBefore],axis = 1)

    afterencode = encafter.transform(df_CS_feed.word_after.values.reshape(-1,1)).toarray()
    dfOneHotAfter = pd.DataFrame(afterencode,columns = ['after_'+str(int(i)) for i in range (afterencode.shape[1])])
    dfOneHotAfter.reset_index(drop=True,inplace=True)
    df_CS_feed = pd.concat([df_CS_feed,dfOneHotAfter],axis = 1)

    X = df_CS_feed.drop(['text','RemovedStopwords','Lemmatized','names','word_before','word_after'],axis = 1)

    y_predict = clf.predict(X)

    # Make predictions
    y_predict_df = pd.DataFrame(y_predict,columns =['ceo_labels'])
    df_final_ceo = pd.concat([df_CS_feed,y_predict_df],axis=1)

    df_final = df_final_ceo[df_final_ceo['ceo_labels']==1][['text','names','ceo_labels']]
    df_final.reset_index(drop=True,inplace=True)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(df_final.text)

    # Grab bigrams of keywords in question
    keyword_bigram = list(nltk.bigrams(basic_kw))
    # convert setences into bigrams
    df_final['sentence_bigrams'] = df_final['text'].apply(lambda x:senttobigram(x))
    df_final['bigrammatchscore'] = df_final['sentence_bigrams'].apply(lambda x: bigrammatches(keyword_bigram,x))
    
    # Grab index of keywords in feature list, compute sum of tfidf of keywords for each sentence 
    kwindex = []
    for i in range(len(kw)):
        if kw[i].lower() in vectorizer.get_feature_names():
            kwindex.append(vectorizer.get_feature_names().index(kw[i].lower()))

    # Grab index of features that are not keywords
    allindex = list(range(len(vectorizer.get_feature_names())))
    notkwindex = [x for x in allindex if x not in basic_kw] #Only use basickw here so that extended list won't have large weight
    
    #Compute sum of tf-idf score for kw for each sentence
    score_kw = [0] * (response.shape[0])
    score_not_kw = [0] * (response.shape[0])
    for i in range(response.shape[0]):
        # Score of keywords
        score_kw[i] = response[i,kwindex].sum(axis=1).tolist()[0][0]
        # Score of non-keywords
        score_not_kw[i] = response[i,notkwindex].sum(axis=1).tolist()[0][0]

    import operator
    df_final['score'] = list(map(operator.add, list(df_final['bigrammatchscore']), score_kw))
    ### Not subtracting score of not keywords because it doesn't improve results
    #df_final['score'] = list(map(operator.sub,list(df_final['score']),score_not_kw))
    
    df_final.sort_values(by = 'score',ascending=False)


# In[1326]:


if WH_word == 'Which':
    df_CS['names'] = df_CS['text'].apply(lambda x: findorg(x))
    s = df_CS[df_CS['names'].map(lambda d: len(d)) > 0]
    s.reset_index(drop = True,inplace =True)
    res = s.set_index(['text'])['names'].apply(pd.Series).stack()
    res = res.reset_index()
    res.drop(['level_1'],axis = 1,inplace = True)
    res.columns = ['text','names']
    res['name_length'] = res['names'].apply(lambda x: len(x))
    res = res[res['name_length'] > 2]
    res.reset_index(drop=True,inplace=True)
    res['RemovedStopwords'] = res['text'].apply(stopwordremoval).apply(joinlist)
    res['Lemmatized'] = res['RemovedStopwords'].apply(lemmatization).apply(joinlist)
    res['Lemmatized'] = res['Lemmatized'].apply(lambda x: removepunc(x))
    res['names'] = res['names'].apply(lambda x: removepunc(x))
    res['word_before'] = res.apply(lambda x : grab_word_before(x['Lemmatized'],x['names']),axis=1)
    res['word_after'] = res.apply(lambda x : grab_word_after(x['Lemmatized'],x['names']),axis=1)
    df_CS_feed = res.copy(deep=True)
    df_CS_feed = df_CS_feed[df_CS_feed['word_after'].isnull() == False]
    df_CS_feed.reset_index(drop = True,inplace = True)
    # Extract Name Features
    df_CS_feed['Capitalized'] = False
    df_CS_feed['lengthofname'] = 0
    df_CS_feed['lengthoftoken'] = 0
    df_CS_feed['nameinbeg'] = False
    df_CS_feed['nameinend'] = False
    df_CS_feed['wordcontaincompanyindicator'] = False

    df_CS_feed['Capitalized'] = df_CS_feed['names'].apply(lambda x: x.istitle())
    df_CS_feed['lengthofname'] = df_CS_feed['names'].apply(lambda x: len(x))
    df_CS_feed['lengthoftoken'] = df_CS_feed['names'].apply(lambda x: len(word_tokenize(x)))
    df_CS_feed['nameinbeg'] = df_CS_feed.apply(lambda x: tokenindex(x['Lemmatized'], word_tokenize(x['names'])[0])==0,axis = 1)
    df_CS_feed['nameinend'] = df_CS_feed.apply(lambda x: tokenindex(x['Lemmatized'], word_tokenize(x['names'])[len(word_tokenize(x['names']))-1]) == len(word_tokenize(x['Lemmatized']))-1,axis = 1)
    df_CS_feed['wordcontaincompanyindicator'] = df_CS_feed['names'].apply(lambda x: hascompanyindicator(x))
    
    # Extract Before and After Word Features
    df_CS_feed['beforewordlength'] = 0
    df_CS_feed['beforewordcapitalized'] = False
    df_CS_feed['beforewordcontainnumbers'] = False
    df_CS_feed['afterwordlength'] = 0
    df_CS_feed['afterwordcapitalized'] = False
    df_CS_feed['afterwordcontainnumbers'] =False
    df_CS_feed['beforewordlength'] = df_CS_feed['word_before'].apply(lambda x: len(x))
    df_CS_feed['beforewordcapitalized'] = df_CS_feed['word_before'].apply(lambda x: x.istitle())
    df_CS_feed['beforewordcontainnumbers'] = df_CS_feed['word_before'].apply(lambda x: hasNumbers(x))
    df_CS_feed['afterwordlength'] = df_CS_feed['word_after'].apply(lambda x: len(x))
    df_CS_feed['afterwordcapitalized'] = df_CS_feed['word_after'].apply(lambda x: x.istitle())
    df_CS_feed['afterwordcontainnumbers'] =df_CS_feed['word_after'].apply(lambda x: hasNumbers(x))


    # Extract Sentence Feature
    df_CS_feed['senetencecontainceo'] = df_CS_feed['Lemmatized'].apply(lambda x: sentencehasceoindicator(x))
    df_CS_feed['senetencecontainbusinesswords'] = df_CS_feed['Lemmatized'].apply(lambda x: havebusinesswords(x))
    
    # Stem words before and after to feed in getdummy
    df_CS_feed['word_before'] = df_CS_feed['word_before'].apply(stemmer).apply(joinlist)
    df_CS_feed['word_after'] = df_CS_feed['word_after'].apply(stemmer).apply(joinlist)

    # Load classifiers and onehotencoder
    from sklearn.preprocessing import OneHotEncoder
    from joblib import dump, load
    clf = joblib.load('companytrainer.joblib')
    logmodel = joblib.load('companylogmodeltrainer.joblib')
    encbefore = joblib.load('companybeforeencode.joblib')
    encafter = joblib.load('companyafterencode.joblib')
    
    beforeencode = encbefore.transform(df_CS_feed.word_before.values.reshape(-1,1)).toarray()
    dfOneHotBefore = pd.DataFrame(beforeencode,columns = ['before_'+str(int(i)) for i in range (beforeencode.shape[1])])
    dfOneHotBefore.reset_index(drop=True,inplace=True)
    df_CS_feed = pd.concat([df_CS_feed,dfOneHotBefore],axis = 1)

    afterencode = encafter.transform(df_CS_feed.word_after.values.reshape(-1,1)).toarray()
    dfOneHotAfter = pd.DataFrame(afterencode,columns = ['after_'+str(int(i)) for i in range (afterencode.shape[1])])
    dfOneHotAfter.reset_index(drop=True,inplace=True)
    df_CS_feed = pd.concat([df_CS_feed,dfOneHotAfter],axis = 1)

    X = df_CS_feed.drop(['text','RemovedStopwords','Lemmatized','names','word_before','word_after','name_length'],axis = 1)
    
    #y_predict = clf.predict(X)
    y_predict = logmodel.predict(X)

    # Make predictions
    y_predict_df = pd.DataFrame(y_predict,columns =['company_labels'])
    df_final_company = pd.concat([df_CS_feed,y_predict_df],axis=1)

    df_final = df_final_company[df_final_company['company_labels']==1][['text','names','company_labels']]
    df_final.reset_index(drop=True,inplace=True)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(df_final.text)

    # Grab bigrams of keywords in question
    keyword_bigram = list(nltk.bigrams(basic_kw))
    # convert setences into bigrams
    df_final['sentence_bigrams'] = df_final['text'].apply(lambda x:senttobigram(x))
    df_final['bigrammatchscore'] = df_final['sentence_bigrams'].apply(lambda x: bigrammatches(keyword_bigram,x))
    
    # Grab index of keywords in feature list, compute sum of tfidf of keywords for each sentence 
    kwindex = []
    for i in range(len(kw)):
        if kw[i].lower() in vectorizer.get_feature_names():
            kwindex.append(vectorizer.get_feature_names().index(kw[i].lower()))

    # Grab index of features that are not keywords
    allindex = list(range(len(vectorizer.get_feature_names())))
    notkwindex = [x for x in allindex if x not in basic_kw] #Only use basickw here so that extended list won't have large weight

    # Grab "bankrupt or bankruptcy" index for each sentence
    bankruptkeywords = ['bankrupt','bankruptcy']
    bankruptkeywordindex = []
    score_bankrupt = [0] * (response.shape[0])
    for i in range(len(bankruptkeywords)):
        if bankruptkeywords[i].lower() in vectorizer.get_feature_names():
            bankruptkeywordindex.append(vectorizer.get_feature_names().index(bankruptkeywords[i].lower()))
    
    #Compute sum of tf-idf score for kw for each sentence
    score_kw = [0] * (response.shape[0])
    score_not_kw = [0] * (response.shape[0])
    for i in range(response.shape[0]):
        # Score of keywords
        score_kw[i] = response[i,kwindex].sum(axis=1).tolist()[0][0]
        # Score of non-keywords
        score_not_kw[i] = response[i,notkwindex].sum(axis=1).tolist()[0][0]
        # Score of bankrupt
        score_bankrupt[i] = response[i,bankruptkeywordindex].sum(axis=1).tolist()[0][0]
    
    df_final['score'] = list(map(operator.add, list(df_final['bigrammatchscore']), score_kw))
    #Not subtracting score of non-keywords because it doesn't work well.
    #df_final['score'] = list(map(operator.sub,list(df_final['score']),score_not_kw))
    df_final['score_bankrupt'] = score_bankrupt
    df_final = df_final[df_final['score_bankrupt']!=0].sort_values(by = 'score',ascending=False)


# In[1348]:


print('Top 10 Possible Answers')
if WH_word == "What":
    if question == 'What affects GDP?':
        print(df_final.sort_values(by = 'score',ascending=False).head(10).names)
    else:
        print(df_final.sort_values(by = 'score',ascending=False).head(10).percentages)
else:
    print(df_final.sort_values(by = 'score',ascending=False).head(10).names)

