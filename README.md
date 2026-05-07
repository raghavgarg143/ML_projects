# ML_projects

Objective: To analyze the customer’s behavior by analyzing customer’s demographics and reviews submitted on the website.

## Loading the Dataset

```jsx
data = pd.read_excel('<file_path.csv>')
data.head()
```

## Data Cleaning

```python
#Checking for the null values
data.isna().sum()

#Removing the null values from 'Review Text' column
data.dropna(subset=['Review Text'],inplace=True)
data = data.reset_index(drop=True)
```

## EDA

#### Plotting the Frequency Distributions

1. **Ratings Distribution**

```python
import seaborn as sns

sns.countplot(x=data['Rating'],data=data)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.show()
```

1. **Recommend Flag Distribution**

```python
sns.countplot(x=data['Recommend Flag'],data=data)
plt.xlabel('Flag')
plt.ylabel('Frequency')
plt.title('Flag Distribution')
plt.show()
```

1. **Age Distribution**

```python
sns.histplot(x=data['Customer Age'],bins=30)  //continuous variable
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()
```

Note: **countplot()** is used to plot categorical values whereas **histplot()** is used to plot continuous values.

## Text Pre-processing

#### Cleaning Function

```python
!pip install spacy
import subprocess
subprocess.run(['python','-m','spacy','download','en_core_web_sm'])

import spacy
nlp = spacy.load('en_core_web_sm')

#Adding a new column as 'clean_text'
def clean_text(text):
		doc=nlp(text)
		clean_text = ""
		special_characters = "+@$%^&*()"
		for token in doc:
				if(not token.is_stop) and (not token.is_punct) and (str(token) not in 
				special_characters) and (token.is_alpha):
						if(len(str(token.lemma_))>2):
							clean_text = clean_text + " " + str(token.lemma_).lower()
		return clean_text.strip()
		
data['clean_text'] = data['Review Text'].apply(clean_text) 
	
```

## Word Frequency Analysis

```python
from collections import Counter

all_words = " ".join(data['clean_text'])
doc = nlp(all_words)
words = [token.text for token in doc]
#words = [token.text for token in doc if not token.is_stop and not token.is_punct]
freq = Counter(words)
freq.most_common(20)

#Most frequent words for positive sentiment and negative sentiment

positive_data = data[data['Sentiment'] == 'Positive']
negative_data = data[data['Sentiment'] == 'Negative']

from collections import Counter

# Positive words
positive_words = " ".join(positive_df['clean_text'])
positive_tokens = positive_words.split()
positive_freq = Counter(positive_tokens)

# Negative words
negative_words = " ".join(negative_df['clean_text'])
negative_tokens = negative_words.split()
negative_freq = Counter(negative_tokens)

positive_freq.most_common(20)
negative_freq.most_common(20)

#Creating the Dataframe
import pandas as pd

pos_df = pd.DataFrame(positive_freq.most_common(20), columns=['Word', 'Count'])
neg_df = pd.DataFrame(negative_freq.most_common(20), columns=['Word', 'Count'])

#Plotting the comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Positive
plt.subplot(1,2,1)
plt.barh(pos_df['Word'], pos_df['Count'])
plt.title('Top Positive Words')

# Negative
plt.subplot(1,2,2)
plt.barh(neg_df['Word'], neg_df['Count'])
plt.title('Top Negative Words')

plt.show()
```

## Word Cloud

```python
pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_words = " ".join(data['clean_text'])
wordcloud = WordCloud(width=800, height=400).generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#WordCloud for Positive and Negative words
from wordcloud import WordCloud

# Positive
WordCloud().generate(positive_words)

# Negative
WordCloud().generate(negative_words)
```

## Sentiment Category Rule

```python
def sentiment_label(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

data['Sentiment'] = data['Rating'].apply(sentiment_label)
```

#### Plotting the Sentiment Ratings

```python
import seaborn as sns

sns.countplot(x='Sentiment', data=data)
plt.show()
```

#### Sentiments by different categories/sub-categories/products by location & age group

```python
df['Sentiment'].value_counts()

#Sentiment by Category
category_sentiment = pd.crosstab(df['Category'], df['Sentiment'], normalize='index') * 100
category_sentiment

category_sentiment.plot(kind='bar', stacked=True, figsize=(10,6))

plt.title('Sentiment Distribution by Category')
plt.ylabel('Percentage')
plt.show()

#Sentiment by Sub-category
subcat_sentiment = pd.crosstab(df['SubCategory1'], df['Sentiment'], normalize='index') * 100

#Sentiment by Location
location_sentiment = pd.crosstab(df['Location'], df['Sentiment'], normalize='index') * 100

#Sentiment by Age

bins = [0, 20, 30, 40, 50, 60, 70]
labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70']
df['Age Group'] = pd.cut(df['Customer Age'], bins=bins, labels=labels)

age_sentiment = pd.crosstab(
								df['Age Group'],
								df['Sentiment'],
								normalize='index') * 100
			
#Plotting the age_sentiment								
age_sentiment.plot(kind='bar', stacked=True, figsize=(10,6))

plt.title('Sentiment by Age Group')
plt.show()

#Sentiment by Channel
channel_sentiment = pd.crosstab(
										df['Channel'],
										df['Sentiment'],
										normalize='index') * 100
```

## Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['Recommend Flag']
```

#### Memory-Efficient Approach

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

tfidf_vec = TfidfVectorizer(
						max_features = 1000,
						ngram_range=(1,2),
						dtype=np.float32,
						analyzer='word'
)

X = tfidf_vec.fit_transform(data['clean_text']) //.toarray() is not required
y = data['<target_variable>']

Note: Sparse matrix stays in memory efficiently
```

## Model Building

#### Splitting the Train and Test data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)
```

#### Fitting the Model

```python
rnd_clif = RandomForestClassifier()
rnd_clif.fit(X_train,y_train)

#Usually LogisticRegression works better with TfidfVectorizer

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model_fit = model.fit(X_train,y_train)
```

#### Predicting the values for Test data

```python
y_pred = rnd_clif.predict(X_test)

#using model_fit
y_pred = model_fit.predict(X_test)
```

#### Evaluating the Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

test_acc = accuracy_score(y_pred, y_test)
****
#Plotting the Classification Report
classification_report(y_pred, y_test)
```

## Topic Modelling

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

words = tfidf.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"Topic {i}:")
    print([words[i] for i in topic.argsort()[-10:]])
```

## NER (Named Entity Recognition)

```python
sample_text = data['Review Text'][0]
doc = nlp(sample_text)
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Predicting on the test data

```python
X_test = tfidf_vec.transform(test['<Review Text>'])
test['prediction'] = model_fit.predict(X_test)
test['prediction_proba'] = model_fit.predict_proba(X_test)[:,1]
```
