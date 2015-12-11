# cs281-brandmentions

A repository for the code and data used in my project for the Harvard course CS281 Advanced Machine Learning. [See the report](https://github.com/jonathanhfriedman/cs281-brandmentions/blob/master/report.pdf) for more details.

Code implementing data scraping, as well as the model, as described the report is in <code>code</code>:
* <code>scraping.py</code>: Scrapes data from reddit.com using the [Python Reddit API Wrapper](https://github.com/praw-dev/praw) (PRAW). If you run this, the data will be different from what I scraped, since the top posts will have changed.
* <code>generate_features.py</code>: Parses the scraped data, as well as the list of brands, to generate the input matrix **X** and the output matrix **Y**. It can restrict these to any number of the most popular brands.
* <code>mrots.py</code>: Implements the multiple regression with output and task structures (MROTS) as proposed in [(Rai et. al., 2012)](http://people.duke.edu/~pr73/recent/morNIPS12.pdf), including the alternating optimization procedure.
* <code>driver.py</code>: Applies MROTS to the problem of predicting brand mentions and compares it with a few baselines. As written, this takes a long time to run, and if just testing it out, you may want to decrease the maxiumum number of iterations and/or increase the tolerance.

The scraped data and most popular brands list are in <code>data</code>.
* <code>mfadata.txt</code>: Scraped post and comment data from reddit.com/r/malefashionadvice. Originally scraped on 10/28/2015, this is the dataset referenced in the report.
* <code>brands.txt</code>: List of popular brands on /r/malefashionadvice along with common misspellings and signature items of each brand. Likely out of date as soon as it's written.