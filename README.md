# Filtering-Trustworthy-Google-Reviews-with-Naive-Bayes-Natural-Language-Processing

## <b><u>Project Overview</u></b>
### <b>Inspiration</b>
With information being shared widely over the internet such as Google reviews, how do we determine whether it is legitimate or it is just a noise? Our team thus wanted to create a system where it can help online users differentiate the valid reviews from those that are promotional and irrelevant. 

### <b>What it does</b>
Our model is designed to analyze and differentiate online reviews, allowing user to identify which will be truly useful to them. It can detect genuine reviews about businesses which often reflects real experiences and opinions on the business. It is also able to identify advertisements such as those that include promotional links which are not helpful to online users. It can also reduce misleading reviews such as those written by those without first-hand visit which will be classified under "rant without visit". Our model also provides a probability confidence scoring, providing user transparency on the certainty of the predicted classification of the review.

### <b>How we built it</b>
We started with a labeled dataset covering different review types: real experiences, ads, irrelevant chatter, and hearsay. After cleaning and preparing the text, we applied NLP techniques like TF-IDF to turn words into meaningful patterns the model could understand.

To enhance this, we integrated the Bayesian network with NLP, allowing us to not only capture the importance of words but also the likelihood of certain features occurring together—such as URLs signaling ads or the mention of business names indicating relevance. This integration gave our model the ability to handle uncertainty and weigh different clues more effectively before making a decision.

### <b>Challenges We Faced</b>
The toughest part was dealing with unbalanced data—some review classification types were rare, so the model struggled to learn those patterns. Another challenge was teaching the model to recognize subtle signals that tell a true customer story apart from hearsay or an ad. 

### <b>Accomplishments that we're proud of</b>
Creating a system that feels intuitive by mixing NLP with Bayesian reasoning
Effectively spotting promotional content using URL detection combined with probabilistic rules
Building a simple interface for users to test and see the classifications in action
Improving our understanding of both machine learning and the problem of fake or irrelevant reviews

### <b>What we learned</b>
Through this project, we discovered the complexity of interpreting individuals' reviews as ideas can be expressed in multiple way and they vary from person to person. We gain insights on how we can make use of NLP to turn those texts into meaningful features that computers are able to analyze. This project also helped strengthened our skills in model training and deploying interactive tools that have significant impact.

## <b><u>Setup</u></b>
#### Step 1: run ```pip install -r requirements.txt```
#### Step 2: run ```py main.py ```
#### Step 3: Click browser link
#### Step 4: Have fun!!! 😀😀😀😀

## <b><u>How to reproduce results</u></b>
### <b>Advertisement</b>
<table border='1'>
  <tr>
    <td>Verified Seller ✅ More than 100 brands available. Lowest price guaranteed. DM now for orders! @jimmy</td>
  </tr>
  <tr>
    <td>Discover exclusive collections from top designers and local artisans. <br><br> Up to 50% off seasonal apparel – Shop Now <br><br> Personalized styling consultations – Book Here <br><br> Buy 2, get 1 free on select accessories – Grab Deal <br><br> From chic dresses to casual wear, LuxeMart ensures your wardrobe is always on-trend. Don’t miss our members-only fashion previews – Join LuxeClub today.</td>
  </tr>
</table>

### <b>Rant without visit</b>
<table border='1'>
  <tr>
    <td>I heard from my friend’s cousin that the food here makes you sick, so I’ll never go.</td>
  </tr>
  <tr>
    <td>From the reviews I read, this place is the worst ever. Won't ever visit it</td>
  </tr>
</table>

### <b>Irrelevant</b>
<table border='1'>
  <tr>
    <td>I’m learning how to code in Python these days, it’s fun.</td>
  </tr>
  <tr>
    <td>Bought my laptop here last week, works perfectly fine so far. Price is reasonable.</td>
  </tr>
</table>