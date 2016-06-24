import glob
import re
import collections
import os
import cPickle as pickle


class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self):
        sources = {'test-pos': '../resource/test/pos', 'test-neg': '../resource/test/neg',
                   'train-pos': '../resource/train/pos', 'train-neg': '../resource/train/neg'}

        outs = {'test-pos': 'parsed-test-pos', 'test-neg': 'parsed-test-neg', 'train-pos': 'parsed-train-pos',
                'train-neg': 'parsed-train-neg'}
        self.sources = sources
        self.train_pos_desc,self.train_pos_scores = parse_scored_files(sources['train-pos'],outs['train-pos'])
        self.train_neg_desc, self.train_neg_scores = parse_scored_files(sources['train-neg'],outs['train-neg'])
        self.test_pos_desc, self.test_pos_scores = parse_scored_files(sources['test-pos'],outs['test-pos'])
        self.test_neg_desc, self.test_neg_scores = parse_scored_files(sources['test-neg'],outs['test-neg'])


def parse_scored_files(dir_name, outfile):
        score_path = "../resource/"+outfile + ".score"
        desc_path = "../resource/"+outfile + ".desc"
        if os.path.isfile(score_path) and os.path.isfile(desc_path):
            with open(score_path, "rb") as fscore:
                try:
                    with open(desc_path, "rb") as fdesc:
                        all_scores = pickle.load(fscore)
                        all_desc = pickle.load(fdesc)
                        return all_desc, all_scores
                except StandardError:
                    pass
        train_files = glob.glob(dir_name + "/*.txt")
        all_desc = []
        all_scores = []
        for flnm in train_files:
            mtch = re.search('(\d+)_(\d+).txt', flnm)
            scr = int(mtch.group(2))
            file = open(flnm, 'r')
            content = file.read().replace('\n', '').lower()
            content = re.sub(r'([^\s\w]|_|\d)+', ' ', content)
            content = re.sub(r'\s+', ' ', content)

            all_scores.append( scr )
            all_desc.append( content )

        pickle.dump(all_scores,open("../resource/"+outfile + ".score", "w"))
        pickle.dump(all_scores, open("../resource/"+outfile + ".desc", "w"))

        print "Finished " + outfile
        return all_desc, all_scores

class Batcher:
    """
    a helper class to create batches given a dataset
    """
    def __init__(self, X, y, batchsize=50):
        """

        :param X: array(any) : array of whole training inputs
        :param y: array(any) : array of correct training labels
        :param batchsize: integer : default = 50,
        :return: self
        """
        self.X = X
        self.y = y
        self.iterator = 0
        self.batchsize = batchsize

    def next_batch(self):
        """
        return the next training batch
        :return: the next batch inform of a tuple (input, label)
        """
        start = self.iterator
        end = self.iterator+self.batchsize
        self.iterator = end if end < len(self.X) else 0
        return self.X[start:end], self.y[start:end]

    @staticmethod
    def chunks(l, n):
        """
        Yield successive n-sized chunks from l.
        :param l: array
        :param n: chunk size
        :return: array of arrays
        """
        r = []
        for i in xrange(0, len(l), n):
            r.append(l[i:i+n])

        return r