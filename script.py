import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_pickle('data/articles.pkl')

tfidf = TfidfVectorizer(stop_words='english', max_features=5000, norm='l2')
weights = tfidf.fit_transform(df['content']).todense()
feat_names = np.array(tfidf.get_feature_names())

class NMF_(object):
    def __init__(self,V,k,iterations):
        self.V = V
        self.k = k
        self.iter = iterations

    def fit(self):
        self.n = self.V.shape[0]
        self.m = self.V.shape[1]
        max_v = np.max(self.V)
        self.W = np.zeros((self.n, self.k))
        for i in xrange(self.n):
            for j in xrange(self.k):
                self.W[i][j] = max_v * np.random.random()
        self.H = np.zeros((self.k, self.m))
        for i in xrange(self.k):
            for j in xrange(self.m):
                self.H[i][j] = max_v * np.random.random()

    def update_H(self):
        self.H = np.linalg.lstsq(self.W, self.V)[0]
        self.H = np.array(np.clip(self.H, 0, np.max(self.H)))

    def update_W(self):
        self.W = (np.linalg.lstsq(self.H.T, self.V.T)[0]).T
        self.W = np.array(np.clip(self.W, 0, np.max(self.W)))

    def fit_transform(self):
        self.fit()
        for i in range(self.iter):
            self.update_H()
            self.update_W()

    def mean_squared(self):
        return np.mean(np.square(self.V - (self.W.dot(self.H))))

    def common_words(self, feat_names):
        word_sort = np.argsort(self.H,axis=1)[:,-10:]
        for topic in xrange(self.k):
            print 'Topic %d:' % topic
            print feat_names[word_sort[topic]]
        # for row in self.H:
        #     print feat_names[np.argsort(row)[-5:]]


skl_nmf = NMF(n_components=10, max_iter=500)
skl_nmf.fit(weights)
components = skl_nmf.components_
W = skl_nmf.fit_transform(weights)
latent_feat = ['US Politics','Middle East','Yankees','World','Econ','Racing','Football','Healthcare','Courts','Arts']

word_sort = np.argsort(W,axis=1)
for article in xrange(10):
    print 'Article: %s' %(df.headline[article])
    print latent_feat
    print W[article]
    print 'Section: %s' % (df.section_name[article])
    



