import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset





# You'll need the WordEmbeddings Class . This class wraps a matrix of word vectors and an Indexer
# in order to index new words. The Indexer contains two special tokens: PAD (index 0) and UNK
# (index 1). You’ll want to use get initialized embedding layer to get a torch.nn.Embedding
# layer that can be used in your network. This layer is trainable (if you set frozen to False), but is
# initialized with the pre-trained embeddings.


class SentimentDatasetDAN(Dataset):
    def __init__(self,infile,word_embeddings,sentence_len):
        self.examples = read_sentiment_examples(infile)
        self.sentence_len = sentence_len
        self.sentences_indices = []
        self.labels = []
        # prep indexer
        indexer = word_embeddings.word_indexer
        # prep unk_idx and pad_idx
        unk_idx = indexer.index_of("UNK")
        pad_idx = indexer.index_of("PAD")
        
        # find id for each word
        for ex in self.examples:
            self.labels.append(ex.label)
            indices = []
            for word in ex.words:
                idx = indexer.index_of(word)
                if idx == -1 : 
                    idx = unk_idx
                indices.append(idx)

            # make sure consistent length
            if len(indices) >self.sentence_len:
                indices = indices[:self.sentence_len]
            else:
                num_pads = self.sentence_len-len(indices)
                indices.extend([pad_idx]*num_pads)

            self.sentences_indices.append(indices)

        
    def __len__(self):
        # return len of total examples
        return len(self.examples)

    def __getitem__(self,idx):
        # return tensor
        return torch.tensor(self.sentences_indices[idx]),torch.tensor(self.labels[idx])
        


class BPEDatasetDAN(Dataset):
    def __init__(self, infile, bpe_tokenizer, sentence_len):
        self.examples = read_sentiment_examples(infile)
        self.sentence_len = sentence_len
        self.bpe = bpe_tokenizer
        
        self.sentences_indices = []
        self.labels = []
        
        for ex in self.examples:
            self.labels.append(ex.label)
            
            # Join words into text and encode with BPE
            text = " ".join(ex.words)
            indices = self.bpe.encode(text)
            
            # just use 0 for B
            if len(indices) < sentence_len:
                indices += [0] * (sentence_len - len(indices))
            else:
                # Truncate
                indices = indices[:sentence_len]
            
            self.sentences_indices.append(indices)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.sentences_indices[idx]),
            torch.tensor(self.labels[idx])
        )




class DAN(nn.Module):
    
    def __init__(
                self,embeddings=None,
                n_class=2,
                n_hidden=100,
                n_layers = 2,
                from_pretrained=True,
                embed_dim=None,
                dropout = 0.25,
                vocab_size=None
                ):
        
        super(DAN,self).__init__()

        self.n_hidden = n_hidden
        self.n_class = n_class
        self.dropout = dropout

        if embed_dim is None:
            if embeddings is not None:
                self.embed_dim = embeddings.get_embedding_length()
            else:
                raise ValueError("Must provide embed_dim when embeddings is None")
        else:
            self.embed_dim = embed_dim

        if from_pretrained:
            if embeddings is None:
                raise ValueError("Cannot use from_pretrained=True without embeddings")
            self.embeddings = embeddings.get_initialized_embedding_layer(frozen = True)
        
        else:
            if vocab_size is not None:
                actual_vocab_size = vocab_size
            elif embeddings is not None:
                actual_vocab_size = embeddings.get_vocab_size()
            else:
                raise ValueError("Must provide vocab_size when embeddings is None")
            self.embeddings = nn.Embedding(num_embeddings = actual_vocab_size,  embedding_dim = self.embed_dim)


        layers = []

        layers.append(nn.Linear(self.embed_dim,self.n_hidden))
        layers.append(nn.ReLU())

        for _ in range(n_layers-1):
            layers.append(nn.Linear(self.n_hidden,self.n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.n_hidden,self.n_class))

        self.classifier = nn.Sequential(*layers)

        self.log_softmax = nn.LogSoftmax(dim=1)



    def forward(self,x):
        x = self.embeddings(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        x = self.log_softmax(x)

        return x 


