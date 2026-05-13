import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
from itertools import chain

class Solution:

    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True) 

        positive_split = [s.split(" ") for s in positive]
        negative_split = [s.split(" ") for s in negative]

        index_to_word = sorted({w for l in chain(negative_split, positive_split) for w in l})
        word_to_index = {value: float(idx + 1) for idx, value in enumerate(index_to_word)}

        tokenised = []
        for p_list in positive_split:
            tokens = []
            for w in p_list:
                tokens.append(word_to_index[w])
            tokenised.append(torch.tensor(tokens, dtype=torch.float32))
        for n_list in negative_split:
            tokens = []
            for w in n_list:
                tokens.append(word_to_index[w])
            tokenised.append(torch.tensor(tokens, dtype=torch.float32))
        
        padded = nn.utils.rnn.pad_sequence(tokenised, batch_first=True)
        return padded
    


        









