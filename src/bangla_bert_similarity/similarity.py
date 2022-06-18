import torch
import transformers
from normalizer import normalize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def score(sentence_1 : str, sentence_2 : str, max_seq = 512, similarity_method = 'cosine'):
    '''
        Take two Bengali sentences as first two arguments. 
        Maximum sequence lenth as third argument. Can be <= 512. By default 512.
        Similarity method for fourth argument. Can be 'cosine' or 'euclidean'. By default 'cosine'.

        Returns similarity score between two sentence. If similarity method is 'euclidean' then lower score means higher similarity.
        If similarity method is 'cosine' then higher score means higher similarity.
    '''
    max_seq_length = max_seq
    sim_method = similarity_method
    
    sentences = [
        sentence_1,
        sentence_2
    ]
    
    # initialize tokenizer and model for bengali
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
    
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    
    for sentence in sentences:
        # this normalization step is required before tokenizing the text
        sentence = normalize(sentence)
        # encode each sentence and append to dictionary
        new_tokens = tokenizer(sentence,
                           max_length=max_seq_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])  # shape of (number of sentences, max_seq_length)
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])    # shape of (number of sentences, max_seq_length)
    
    # process the tokens through the model
    outputs = model(**tokens)
    # the dense vector representations of the input text which could be accessed through 'last_hidden_state'
    embeddings = outputs.last_hidden_state  # shape of (number of sentences, max_seq_length, hidden size)
    
    # for mean pooling operation, multiply each value in the embeddings tensor by its respective 'attention_mask' value — 
    # so that non-real tokens could be ignored
    # resize the 'attention_mask' tensor
    attention_mask = tokens['attention_mask']   # shape of (number of sentences, max_seq_length)
    # shape of (number of sentences, max_seq_length, hidden size)
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # multiply the two tensors to apply the attention mask
    masked_embeddings = embeddings * mask   # shape of (number of sentences, max_seq_length, hidden size)
    # sum the remaining of the embeddings along axis 1
    summed = torch.sum(masked_embeddings, 1)    # shape of (number of sentences, hidden size)
    # sum the number of values that must be given attention in each position of the tensor
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)    # shape of (number of sentences, hidden size)
    # calculate the mean as the sum of the embedding activations 'summed' divided by the number of values that should be 
    # given attention in each position 'summed_mask'
    mean_pooled = summed / summed_mask  # shape of (number of sentences, hidden size)
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    
    # calculate the similarity for sentences[0] with the rest    
    if sim_method == "euclidean":                                               # lower is better; highly similar
        sim_result = euclidean_distances([mean_pooled[0]],mean_pooled[1:])   
    elif sim_method == "cosine":                                                # higher is better; highly similar
        sim_result = cosine_similarity([mean_pooled[0]],mean_pooled[1:])
    
    return sim_result[0]   


def scores(reference:str, measure_sentences:list, max_seq = 512, similarity_method = 'cosine'):
    '''
        Take two Bengali sentences as first two arguments. 
        Maximum sequence lenth as third argument. Can be <= 512. By default 512.
        Similarity method for fourth argument. Can be 'cosine' or 'euclidean'. By default 'cosine'.

        Returns similarity score between two sentence. If similarity method is 'euclidean' then lower score means higher similarity.
        If similarity method is 'cosine' then higher score means higher similarity.
    '''
    max_seq_length = max_seq
    sim_method = similarity_method
    
    sentences = []
    sentences.append(reference)

    for sentence in measure_sentences:
        sentences.append(sentence)
    
    # initialize tokenizer and model for bengali
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
    
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    
    for sentence in sentences:
        # this normalization step is required before tokenizing the text
        sentence = normalize(sentence)
        # encode each sentence and append to dictionary
        new_tokens = tokenizer(sentence,
                           max_length=max_seq_length,
                           truncation=True,
                           padding='max_length',
                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])  # shape of (number of sentences, max_seq_length)
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])    # shape of (number of sentences, max_seq_length)
    
    # process the tokens through the model
    outputs = model(**tokens)
    # the dense vector representations of the input text which could be accessed through 'last_hidden_state'
    embeddings = outputs.last_hidden_state  # shape of (number of sentences, max_seq_length, hidden size)
    
    # for mean pooling operation, multiply each value in the embeddings tensor by its respective 'attention_mask' value — 
    # so that non-real tokens could be ignored
    # resize the 'attention_mask' tensor
    attention_mask = tokens['attention_mask']   # shape of (number of sentences, max_seq_length)
    # shape of (number of sentences, max_seq_length, hidden size)
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # multiply the two tensors to apply the attention mask
    masked_embeddings = embeddings * mask   # shape of (number of sentences, max_seq_length, hidden size)
    # sum the remaining of the embeddings along axis 1
    summed = torch.sum(masked_embeddings, 1)    # shape of (number of sentences, hidden size)
    # sum the number of values that must be given attention in each position of the tensor
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)    # shape of (number of sentences, hidden size)
    # calculate the mean as the sum of the embedding activations 'summed' divided by the number of values that should be 
    # given attention in each position 'summed_mask'
    mean_pooled = summed / summed_mask  # shape of (number of sentences, hidden size)
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    
    # calculate the similarity for sentences[0] with the rest    
    if sim_method == "euclidean":                                               # lower is better; highly similar
        sim_result = euclidean_distances([mean_pooled[0]],mean_pooled[1:])   
    elif sim_method == "cosine":                                                # higher is better; highly similar
        sim_result = cosine_similarity([mean_pooled[0]],mean_pooled[1:])
    
    return sim_result[0]