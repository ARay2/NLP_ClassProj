from sklearn.svm           import SVC
import nltk
from nltk.corpus           import wordnet as wn
from utils.dataset         import Dataset
from collections           import Counter
import pyphen
import spacy
from nltk import pos_tag, word_tokenize


class Features(object):
    def __init__(self, language):

        self.language = language

        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if  self.language == 'english':
            self.avg_word_length = 5.3
            self.syl = pyphen.Pyphen(lang='en')
            self.nlp = spacy.load('en_core_web_lg')

        elif self.language == 'spanish':
            self.avg_word_length = 6.2
            self.syl = pyphen.Pyphen(lang='es')
            self.nlp = spacy.load('es_core_news_md')
 
        self.model = SVC()

    def extract_features(self, word):
        postag = {'CC': 1, 'CD': 2,'DT': 3,'EX': 4,'FW': 5,'IN': 6,'JJ': 7,'JJR': 8,'JJS': 9,'LS': 10,'MD': 11,'NN': 12,'NNS': 13,'NNP': 14,'NNPS': 15,'PDT': 16,'POS': 17,'PRP': 18,'PRP$': 19,'RB': 20,'RBR': 21,'RBS': 22,'RP': 23,'TO': 24,'UH': 25,'VB': 26,'VBD': 27,'VBG': 28,'VBN': 29,'VBP': 30,'VBZ': 31,'WDT': 32,'WP': 33,'WP$': 34,'WRB': 35}
        sent = word.lower()
        ext_feats = []
        ext_feats.append(len(sent)/self.avg_word_length)
        ext_feats.append(len(sent.split()))
        # ext_feats.append(len(self.syl.inserted(sent).split('-')))
        ext_feats.append(len(wn.synsets(sent)))
        count1 = 0
        count2 = 0
        for hyp in wn.synsets(sent):
            count1+=len(hyp.hypernyms())
            count2+=len(hyp.hyponyms())
        ext_feats.append(count1)
        ext_feats.append(count2)    
        ext_feats.append(postag[nltk.pos_tag(word_tokenize(sent))[-1][-1]])
        ext_feats += list(self.nlp(sent).vector)
        return ext_feats


    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'])) 
        return self.model.predict(X)