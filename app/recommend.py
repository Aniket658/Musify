import pandas as pd
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 

nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet') 


lemmatizer = WordNetLemmatizer() 

nltk.download('stopwords') 
stop_words = set(stopwords.words('english')) 

VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}


def cosine_sim(genre, song):
    pth = "C:/Users/sai/Desktop/data/" + genre +".csv"
    df = pd.read_csv(pth)
    df = df.set_index('song')

    df["lyrics_processed"]= df["lyrics"].apply(preprocess_sentences)

    
    tfidfvec = TfidfVectorizer() 
    tfidf_song = tfidfvec.fit_transform((df["lyrics_processed"])) 
    
    cos_sim = cosine_similarity(tfidf_song, tfidf_song)

    indices = pd.Series(df.index) 
    return recommendations(song, cos_sim, indices, df)

def preprocess_sentences(text): 
    text = text.lower() 
    temp_sent =[] 
    words = nltk.word_tokenize(text) 
    tags = nltk.pos_tag(words) 
    for i, word in enumerate(words): 
        if tags[i][1] in VERB_CODES:  
            lemmatized = lemmatizer.lemmatize(word, 'v') 
        else: 
            lemmatized = lemmatizer.lemmatize(word) 
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized) 
            
    finalsent = ' '.join(temp_sent) 
    finalsent = finalsent.replace("n't", " not") 
    finalsent = finalsent.replace("'m", " am") 
    finalsent = finalsent.replace("'s", " is") 
    finalsent = finalsent.replace("'re", " are") 
    finalsent = finalsent.replace("'ll", " will") 
    finalsent = finalsent.replace("'ve", " have") 
    finalsent = finalsent.replace("'d", " would") 
    return finalsent 
    
    
  
def recommendations(song, cosine_sim, indices, df): 
    recommended_songs = [] 
    index = indices[indices == song].index[0] 
    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False) 
    top_5 = list(similarity_scores.iloc[1:min(len(indices),6)].index) 
    for i in top_5: 
        recommended_songs.append(list(df.index)[i]) 
    print("Song : Believer")
    print(recommended_songs)
    return recommended_songs

