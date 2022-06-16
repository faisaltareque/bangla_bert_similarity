from bangla_bert_similarity import calculate_similarity

def test():
    assert calculate_similarity.similarity_score('a','b') == 'ab'