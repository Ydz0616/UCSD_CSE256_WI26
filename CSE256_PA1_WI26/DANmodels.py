import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
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
        
class DAN(nn.Module):
    # two-layer
    def __init__(self,n_hidden_units,word_embeddings,num_classes):
        super(DAN, self).__init__()

        # hyper-params
        self.n_hidden_units = n_hidden_units
        self.num_classes = num_classes
        self.embed_dim = word_embeddings.get_embedding_length()


        # model architecture
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim,self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units,self.num_classes)
        )
        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,input_text):
        
        # get embedding
        data = self.embeddings(input_text)
        # calculate average
        data = data.mean(dim=1)
        #  run this through the classifier
        logits = self.classifier(data)
        return self._log_softmax(logits)



