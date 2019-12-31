# My Book Recommender System

Book Recommedation Website:

http://booksrecommend.ml/

or http://datadong.xyz

- [save_book_title](https://github.com/dongzhang84/BookReco/blob/master/save_book_title.ipynb)

extract useful information from goodreads_books.json.gz

- [load_reviews](https://github.com/dongzhang84/BookReco/blob/master/load_reviews.ipynb)

save DataFrame df_reviews_short (ratings_count > 1000) and combine all reviews for each book

- [review1000.py](https://github.com/dongzhang84/BookReco/blob/master/review1000.py)

from df_reviews_short choose the top 1000 books' information and save as df_reviews1000.csv

- [models](https://github.com/dongzhang84/BookReco/blob/master/models.ipynb) 

generate word embedding models

- [recommender_current](https://github.com/dongzhang84/BookReco/blob/master/recommender_current.ipynb)

build recommender system based on models