from utils.dataset         import Dataset
from utils.array           import Features
from utils.scorer          import report_score
from utils.baseline        import Baseline
import numpy
from pylab import *

for lang in ["english", "spanish"]:
    data = Dataset(lang)
    model = Features(lang)
    # baseline = Baseline(lang)

    # baseline.train(data.trainset)
    # paseline = baseline.test(data.testset)
    
    
    print("{}: {} training - {} dev - {} test".format(lang, len(data.trainset), len(data.devset), len(data.testset)))

    model.train(data.trainset)
    predictions = model.test(data.testset)
    gold_labels = [sent['gold_label'] for sent in data.testset]

    pl = numpy.cumsum([predic == sent['gold_label'] for sent, predic in zip(data.testset, predictions) ]) / range(1,len(data.testset)+1)
    # pl = numpy.cumsum([predic == sent['gold_label'] for sent, predic in zip(data.testset, paseline) ]) / range(1,len(data.testset)+1)

    report_score(gold_labels, predictions)
    # report_score(gold_labels, paseline)
    
    
    plt.title('graph for learning rate')
    plt.plot(100*pl[20:])
    plt.ylabel('accuracy score')
    plt.xlabel('iteration')
    plt.show()
