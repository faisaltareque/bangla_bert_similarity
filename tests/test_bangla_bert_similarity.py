from bangla_bert_similarity import similarity

def test_similarity():
    sentence1 = 'তোমার সাথে দেখা হয়ে ভালো লাগলো।'
    sentence2 = 'আপনার সাথে দেখা হয়ে ভালো লাগলো।'
    score = similarity.score(sentence1, sentence2)
    assert str(score) == '[0.83910286]'