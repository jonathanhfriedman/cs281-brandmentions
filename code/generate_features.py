import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_brands(num_brands):
    with open('brands.txt', 'r') as infile:
        brands = []
        for line in infile:
            brands.append(line.rstrip('\n').split(','))
            
    with open('mfadata.txt', 'r') as infile:
        posts, comments = [[] for i in range(2)]
        splitfile = infile.read().split('\n')
        for s in enumerate(splitfile):
            sp = s[1].split(',')
            posts.append(sp[0])
            for i in range(1, len(sp)):
                comments.append((s[0], sp[i]))
                
    posts = posts[:-1] # the last post is empty
                  
    brandcount = np.zeros((len(posts), len(brands))).astype('float')
    found = False

    for c in comments:
        for b in enumerate(brands):
            found = False
            for j in b[1]:
                if found == False and j in c[1]:
                    brandcount[c[0], b[0]] += 1
                    found = True

    for i in range(brandcount.shape[0]):
        if np.sum(brandcount[i, :]) != 0:
            brandcount[i, :] = brandcount[i, :] / np.sum(brandcount[i, :])

    if num_brands is not None:
        # Construct brandcount with only the num_brands most popular brands
        assert(num_brands <= 74), 'num_brands={}, must be <= 74'.format(num_brands)
        brandpop = np.sum(brandcount, axis=0)
        brand_num = brandcount[:, np.argpartition(brandpop, -num_brands)[-num_brands:]]
        for i in range(brand_num.shape[0]):
            if np.sum(brand_num[i, :]) != 0:
                brand_num[i, :] = brand_num[i, :] / np.sum(brand_num[i, :])
        brandcount = brand_num

    fmat = construct_features(posts)
    return brandcount, fmat

def construct_features(posts):
    """
    Construct a feature matrix with bag of words, average tf-idf, number of words
    per post, and average word length.
    """
    # Bag of words
    vectorizer = CountVectorizer(stop_words='english', max_features=150, binary=True, ngram_range=(1,1))
    binmat = vectorizer.fit_transform(posts)

    # tf-idf
    tvectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    tmat = np.sum(tvectorizer.fit_transform(posts).toarray(), axis=1)
    for i in range(len(posts)):
        tmat[i] = tmat[i] / len(posts[i].split())
            
    fmat = np.concatenate((binmat.toarray(), tmat.reshape(len(tmat), 1)), axis=1)

    # Post length
    fmat = np.concatenate((fmat, np.array([len(p.split()) for p in posts]).reshape(len(posts), 1)), axis=1)

    # Avg word length
    avgword = np.array([1. * len(p.strip()) / len(p.split()) for p in posts]).reshape(len(posts), 1)
    fmat = np.concatenate((fmat, avgword), axis=1)

    return fmat
