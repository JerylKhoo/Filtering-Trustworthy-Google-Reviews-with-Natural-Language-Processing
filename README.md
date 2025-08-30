# Filtering-Trustworthy-Google-Reviews-with-Naive-Bayes-Natural-Language-Processing

## Project Overview
### Inspiration
With information being shared widely over the internet such as Google reviews, how do we determine whether it is legitimate or it is just a noise? Our team thus wanted to create a system where it can help online users differentiate the valid reviews from those that are promotional and irrelevant. 

### What it does
Our model is designed to analyze and differentiate online reviews, allowing user to identify which will be truly useful to them. It can detect genuine reviews about businesses which often reflects real experiences and opinions on the business. It is also able to identify advertisements such as those that include promotional links which are not helpful to online users. It can also reduce misleading reviews such as those written by those without first-hand visit which will be classified under "rant without visit". Our model also provides a probability confidence scoring, providing user transparency on the certainty of the predicted classification of the review.

### How we built it
We started with a labeled dataset covering different review types: real experiences, ads, irrelevant chatter, and hearsay. After cleaning and preparing the text, we applied NLP techniques like TF-IDF to turn words into meaningful patterns the model could understand.

To enhance this, we integrated the Bayesian network with NLP, allowing us to not only capture the importance of words but also the likelihood of certain features occurring together—such as URLs signaling ads or the mention of business names indicating relevance. This integration gave our model the ability to handle uncertainty and weigh different clues more effectively before making a decision.

### Challenges We Faced
The toughest part was dealing with unbalanced data—some review classification types were rare, so the model struggled to learn those patterns. Another challenge was teaching the model to recognize subtle signals that tell a true customer story apart from hearsay or an ad. 

### Accomplishments that we're proud of
Creating a system that feels intuitive by mixing NLP with Bayesian reasoning
Effectively spotting promotional content using URL detection combined with probabilistic rules
Building a simple interface for users to test and see the classifications in action
Improving our understanding of both machine learning and the problem of fake or irrelevant reviews

### What we learned
Through this project, we discovered the complexity of interpreting individuals' reviews as ideas can be expressed in multiple way and they vary from person to person. We gain insights on how we can make use of NLP to turn those texts into meaningful features that computers are able to analyze. This project also helped strengthened our skills in model training and deploying interactive tools that have significant impact.

## <u>Setup</u>
### Step 1: run ```pip install -r requirements.txt```
### Step 2: run ```py main.py ```
### Step 3: Click browser link
### 😀😀😀😀