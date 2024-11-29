# ZeeRecommender

This is movies recommendation system for the data available from Zee.
Two types of recommenders are developed.

#### 1. Pearson Correlation based recommender

In this model item-item similarity is calculated based on pearson correlation coefficient

#### 2. Cosine Similarity based recommender

In this model item-item similarity is calculated based on cosine similarity between two items

# Movie Recommendation

To get the recommended movies for given user ID two variables are given as input in json format to the model. `user` and `n_recommend` of movies to be recommended

Range of `user` is from 1 to 6040
Range of `n_recommend` is from 1 to 50

Movies to the particular users are recommended after calculating predicted ratings for unrated moveis by users followed by selecting `n_recommend` randomly from movies having predicted ratings more than 4.5

Ratings are predicted with the formula

$$ \hat{r}_{ui} = \frac{\sum_{j{\in}u}sim(i,j){\cdot}r_{ui}}{\sum_{j{\in}u}|sim(i,j)|} $$

Where: $N_u$: Items rated by user $u$  
$sim(i,j)$: Similarity between items $i$ and $j$  
$r_{uj}$: Rating by user $u$ for item $j$

# Steps to get prediction

1. Set environment to python 3.13
2. run the flask app with command `flask --app main run`
3. Send a POST request in json format from any request sending platform like POSTMAN
4. Example of sending data in json format `{"user":1, n_recommend:10}`
5. URL for Pearson based recommender is http://localhost:5000/pearson and for Cosine recommender is http://localhost:5000/cosine
