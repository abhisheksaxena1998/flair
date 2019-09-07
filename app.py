#importing libraries
import sklearn
import pickle
import praw
import re
from bs4 import BeautifulSoup
import nltk
# nltk.download('all')
from nltk.corpus import stopwords
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
@app.route('/statistics')
def statistics():
    return flask.render_template('statistics.html')

@app.route("/register", methods=["POST"])
def register():
    if request.method=='POST':
        nm = request.form.get("url")


        reddit = praw.Reddit(client_id='WBTxS7rybznf7Q', client_secret='vJUTUflXITBsQMxeviOfG8mCZoA', user_agent='projectreddit', username='Mysterious_abhE', password='Saxena0705')
        #loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

        filename='Regressor_model_final.pkl'

        loaded_model =pickle.load(open(filename, 'rb'))

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        def clean_text(text):
   
            text = BeautifulSoup(text, "lxml").text
            text = text.lower()
            text = REPLACE_BY_SPACE_RE.sub(' ', text)
            text = BAD_SYMBOLS_RE.sub('', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)
            return text

        """### Detect Reddit India Post Flair"""

        def detect_flair(url,loaded_model):

            submission = reddit.submission(url=url)

            data = {}

            data['title'] = submission.title
            data['url'] = submission.url

            submission.comments.replace_more(limit=None)
            comment = ''
            for top_level_comment in submission.comments:
                comment = comment + ' ' + top_level_comment.body
            data["comment"] = comment
            data['title'] = clean_text(data['title'])
            data['comment'] = clean_text(data['comment'])
            data['combine'] = data['title'] + data['comment'] + data['url']
  
            return loaded_model.predict([data['combine']])
            #print (loaded_model.predict([data['combine']]))

        #url='https://www.reddit.com/r/AskReddit/comments/y3tzl/indians_of_reddit_india_is_in_55th_place_in/'
        #detect_flair(nm,loaded_model)



        return flask.render_template('result.html',prediction=detect_flair(nm,loaded_model),url=nm)

       
