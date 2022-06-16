from bangla_bert_similarity import similarity

def test():
    assert similarity.score('a','b') == 'ab'