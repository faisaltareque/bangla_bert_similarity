from bangla_bert_similarity import similarity_score

def test_similarity():
    sentence1 = 'যোগাযোগ মন্ত্রী বলেছেন এ নিয়ে আমরা আলোচনা করেছি'
    sentence2 = 'মানিকগঞ্জের দৌলতপুর উপজেলা আওয়ামী লীগের ত্রিবার্ষিক সম্মেলন অনুষ্ঠিত হয়েছে'
    score = similarity_score(sentence1, sentence2)
    assert str(score) == '[0.38779554]'
