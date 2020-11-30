"""
reproduce TVC evaluation using pycocoevalcap from Maluuba nlg-eval (Python 3)
"""
import json

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge


def _remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class TVCEval(object):
    """ preload evaluation tools and references for repeated evaluation """
    def __init__(self, ref_path):
        self.tokenizer = PTBTokenizer()
        id2refs = {ex['clip_id']: [_remove_nonascii(cap['desc'].strip())
                                   for cap in ex['descs']]
                   for ex in map(json.loads, open(ref_path))}
        self.id2refs = self.tokenizer.tokenize(id2refs)
        self.scorers = []
        self.scorers.append((Bleu(4),
                             ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        self.scorers.append((Meteor(), "METEOR"))
        self.scorers.append((Rouge(), "ROUGE_L"))
        self.scorers.append((Cider(), "CIDEr"))

    def __call__(self, json_res):
        """ corpus level metrics, take list of results """
        id2hyps = {
            res['clip_id']: [_remove_nonascii(res['descs'][0]['desc'].strip())]
            for res in json_res
        }
        id2hyps = self.tokenizer.tokenize(id2hyps)
        assert len(id2hyps) == len(self.id2refs)

        ret_scores = {}
        for scorer, method in self.scorers:
            print(f"Computing {method} score...")
            score, scores = scorer.compute_score(self.id2refs, id2hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc * 100
            else:
                ret_scores[method] = score * 100

        return ret_scores
