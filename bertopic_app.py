import streamlit as st
import pandas as pd
import numpy as np
import base64
import warnings
warnings.filterwarnings("ignore")
import re
from tqdm import tqdm
from quickstart import get_service, get_data
import contractions
import string
import demoji
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic



def parse_main_reads(date, msg_body):
    """ Returns a list of article information"""
    
    start_str = 'to get The Download every day.\r\n\r\n'
    content_str = msg_body[msg_body.find(start_str) + len(start_str):msg_body.find('We can still have nice things')]
    content_str = re.sub(r"http\S+", "", content_str, flags=re.MULTILINE)
    content_str = re.sub(r" \( ", "", content_str, flags=re.MULTILINE)
    content_str = re.sub(r'\r\n', ' ', content_str, flags=re.MULTILINE)
    articles = content_str.split('------------------------------------------------------------')
    articles = [text[:text.find('Read the full story')] for text in articles]
    return articles

def parse_must_reads(date, msg_body):
    """ Returns a list of article information
        title, subtitle, author, publication, minutes (reading time)"""
    
    must_reads = msg_body[msg_body.index('The must-reads'):]
    text = re.sub(r'\(https?:\S+.*\)', '', msg_body, flags=re.MULTILINE)
    articles = []
    for i in range(1, 11):
        try:
            articles.append(list(re.findall('\n' + str(i) + ' (.*)\r\n(.*) \((.*) \r\n', text, re.MULTILINE)[0]))
        except:
            try:
                articles.append(list(re.findall('\n' + str(i) + ' (.*)\r\n(.*)\((.*) \r\n', text, re.MULTILINE)[0]))
            except:
                continue
    return articles

@st.cache(suppress_st_warning=True)
def load_data():
    service = get_service()
    messages = get_data(service, 'MIT Download')
    my_bar = st.progress(0)
    data = []
    for i, message in enumerate(messages):
        my_bar.progress((i+1) / len(messages))

        # Get an email by id
        msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        
        # Get date of email for the purpose of topic modeling over time
        for date_dict in msg['payload']['headers']:
            if date_dict['name'] == 'Date':
                date = date_dict['value']
        date = pd.to_datetime(date)

        # Get the email body
        content = msg['payload']['parts'][0]['body']['data']
        msg_body = base64.urlsafe_b64decode(content).decode('utf-8')
        
        # Extract article information for all articles featured in daily digest
        if msg_body.find('The must-reads') != -1:
            fetched_articles = parse_must_reads(date, msg_body)
            for must_reads in fetched_articles:
                data.append([date, *must_reads])
        else:
            continue

    return data

def preprocess(text_col):
    """This function will apply NLP preprocessing lambda functions over a pandas series such as df['text'].
       These functions include converting text to lowercase, removing emojis, expanding contractions, removing punctuation,
       removing numbers, removing stopwords, lemmatization, etc."""
    
    # convert to lowercase
    text_col = text_col.apply(lambda x: ' '.join([w.lower() for w in x.split()]))
    
    # remove emojis
    text_col = text_col.apply(lambda x: demoji.replace(x, ""))
    
    # expand contractions  
    text_col = text_col.apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))

    # remove punctuation
    text_col = text_col.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    
    # remove numbers
    text_col = text_col.apply(lambda x: ' '.join(re.sub("[^a-zA-Z]+", " ", x).split()))

    # remove stopwords
    stopwords = [sw for sw in list(nltk.corpus.stopwords.words('english')) + ['thing'] if sw not in ['not']]
    text_col = text_col.apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))

    # lemmatization
    text_col = text_col.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))

    # remove short words
    text_col = text_col.apply(lambda x: ' '.join([w.strip() for w in x.split() if len(w.strip()) >= 3]))

    return text_col


@st.cache
def make_wordcloud(df):
    # change the value to black
    def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl(0,100%, 1%)")
    # set the wordcloud background color to white
    # set max_words to 1000
    # set width and height to higher quality, 3000 x 2000
    wordcloud = WordCloud(font_path = '~/Library/Fonts/IBMPlexSans-Light.ttf', 
                        background_color="white", width=3000, height=2000, collocations=True,
                        max_words=500).generate_from_text(df['text'].to_string())

    # set the word color to black
    wordcloud.recolor(color_func = black_color_func)
    fig = plt.figure(figsize=[15,10])
    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove plot axes
    plt.axis("off")
    return fig

@st.cache(allow_output_mutation=True)
def get_topic_model(df):
    text = df['text'].to_list()
    dates = df['date'].apply(lambda x: pd.Timestamp(x))
    topic_model = BERTopic(min_topic_size=len(text) // 100, n_gram_range=(1,3), verbose=False)
    topics, _ = topic_model.fit_transform(text)
    return text, dates, topic_model, topics


df = None
uploaded_file = st.sidebar.file_uploader('Choose a CSV file')
st.sidebar.caption('Make sure the csv contains a column titled "date" and a column titled "text"')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # st.write(df)
elif st.sidebar.button('Load demo data'):
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data... done!')
    # if st.checkbox('Preview the data'):
    #     st.subheader('5 rows of raw data')
    #     st.write(data[:5])

    df = pd.DataFrame(data, columns = ['date', 'title', 'subtitle', 'source'])
    df['text'] = df[['title', 'subtitle']].agg(' '.join, axis=1)
    st.write(df.head())

if df is not None:
    # concatenate title and subtitle columns
    data_clean_state = st.text('Cleaning data...')
    # df['text'] = preprocess(df['text'])
    cleaned_df = df[['date', 'text']]
    cleaned_df = cleaned_df.dropna(subset=['text'])
    st.write(len(cleaned_df), "total documents")
    data_clean_state.text('Cleaning data... done!')

    tm_state = st.text('Modeling topics...')
    text, dates, topic_model, topics = get_topic_model(cleaned_df)
    tm_state.text('Modeling topics... done!')

    freq = topic_model.get_topic_info(); 
    st.write(freq.head(10))

    # st.write('Take a closer look at frequent topics')
    # topic_num = st.number_input('Select a topic number', min_value=0, max_value=len(freq), value=3)
    # topic_nr = freq.iloc[topic_num]["Topic"]  # We select a frequent topic
    # st.write(topic_model.get_topic(topic_nr))   # You can select a topic number as shown above

    st.write(topic_model.visualize_topics())

    topics_over_time = topic_model.topics_over_time(docs=text, 
                                                    topics=topics, 
                                                    timestamps=dates, 
                                                    global_tuning=True, 
                                                    evolution_tuning=True, 
                                                    nr_bins=len(cleaned_df['date'].unique()) // 7)
    st.write(topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10))

    st.write(topic_model.visualize_barchart(top_n_topics=9, n_words=5, height=800))

    st.write(topic_model.visualize_term_rank())

    new_df = cleaned_df.copy()
    new_df['topic'] = new_df['text'].apply(lambda x: topic_model.find_topics(x)[0][0])
    st.write(new_df)

    # str_input = st.text_input('Enter a word or phrase to find nearest topic: ', value='taliban')
    # st.write(topic_model.find_topics(str_input))
    
    num_input = st.number_input('Enter a topic number: ', value=3, min_value=0, max_value=len(topics))
    st.write(topic_model.get_representative_docs(num_input))