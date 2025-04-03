import os
import math
import json
import random
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

# Dataset Loading
def load_data():
    with open('train.pkl', 'rb') as file:
        train = pickle.load(file)
    with open('test.pkl', 'rb') as file:
        test = pickle.load(file)
    return train, test


train, test = load_data()
# Test Harness
class Tester:
    def __init__(self, predictor, title=None, data=test, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function):
        cls(function).run()


# Feature Engineering
def extract_features(train, test):
    for item in train + test:
        item.features = json.loads(item.details)
    return train, test

def get_features(item):
    def get_weight(item):
        weight_str = item.features.get('Item Weight')
        if weight_str:
            parts = weight_str.split(' ')
            amount = float(parts[0])
            unit = parts[1].lower()
            if unit=="pounds":
                return amount
            elif unit=="ounces":
                return amount / 16
            elif unit=="grams":
                return amount / 453.592
            elif unit=="milligrams":
                return amount / 453592
            elif unit=="kilograms":
                return amount / 0.453592
            elif unit=="hundredths" and parts[2].lower()=="pounds":
                return amount / 100
            else:
                print(weight_str)
        return None

    def get_rank(item):
        rank_dict = item.features.get("Best Sellers Rank")
        if rank_dict:
            ranks = rank_dict.values()
            return sum(ranks)/len(ranks)
        return None

    def get_text_length(item):
        return len(item.test_prompt())

    def is_top_electronics_brand(item):
        TOP_ELECTRONICS_BRANDS = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
        brand = item.features.get("Brand")
        return brand and brand.lower() in TOP_ELECTRONICS_BRANDS
    
    return {
        "weight": get_weight(item),
        "rank": get_rank(item),
        "text_length": get_text_length(item),
        "is_top_electronics_brand": 1 if is_top_electronics_brand(item) else 0
    }

def list_to_dataframe(items):
    features = [get_features(item) for item in items]
    df = pd.DataFrame(features)
    df['price'] = [item.price for item in items]
    return df

# Models
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_bow_model(documents, prices):
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(documents)
    model = LinearRegression()
    model.fit(X, prices)
    return vectorizer, model

def train_word2vec_model(documents):
    processed_docs = [simple_preprocess(doc) for doc in documents]
    model = Word2Vec(sentences=processed_docs, vector_size=400, window=5, min_count=1, workers=8)
    return model

def train_svr(X, y):
    model = LinearSVR()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    model.fit(X, y)
    return model

# Main Function
def main():
    train, test = extract_features(train, test)

    # Feature Engineering
    train_df = list_to_dataframe(train)
    test_df = list_to_dataframe(test[:250])

    # Linear Regression
    feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']
    X_train = train_df[feature_columns]
    y_train = train_df['price']
    X_test = test_df[feature_columns]
    y_test = test_df['price']
    linear_model = train_linear_regression(X_train, y_train)

    # Bag of Words Model
    documents = [item.test_prompt() for item in train]
    prices = np.array([float(item.price) for item in train])
    vectorizer, bow_model = train_bow_model(documents, prices)

    # Word2Vec Model
    w2v_model = train_word2vec_model(documents)
    X_w2v = np.array([np.mean([w2v_model.wv[word] for word in simple_preprocess(doc) if word in w2v_model.wv], axis=0) for doc in documents])
    svr_model = train_svr(X_w2v, prices)
    rf_model = train_random_forest(X_w2v, prices)

    # Testing
    Tester.test(lambda item: linear_model.predict(pd.DataFrame([get_features(item)]))[0])
    Tester.test(lambda item: max(bow_model.predict(vectorizer.transform([item.test_prompt()]))[0], 0))
    Tester.test(lambda item: max(rf_model.predict([np.mean([w2v_model.wv[word] for word in simple_preprocess(item.test_prompt()) if word in w2v_model.wv], axis=0)])[0], 0))

if __name__ == "__main__":
    main()
