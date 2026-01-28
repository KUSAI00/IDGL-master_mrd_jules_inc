import torch
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBertTokenizerWrapper:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def encode(self, text, max_length=None):
        if isinstance(text, list):
            # If input is a list of words, join them
            text = " ".join(text)
        
        # We need to handle truncation manually or rely on tokenizer
        # The existing code handles max_seq_len slicing on words, 
        # but for BERT we should handle it on tokens.
        
        return self.tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

class DistilBertEmbeddings(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', freeze=True):
        super(DistilBertEmbeddings, self).__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None):
        # Return static embeddings
        return self.model.embeddings.word_embeddings(input_ids)

    def get_contextual_embeddings(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = (input_ids != 0).long() # Assuming 0 is padding
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        elif inputs_embeds is not None:
             outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        return outputs.last_hidden_state
