import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id.clear()
        self.id_to_word.clear()
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        str_set = set()
        for text in texts: 
            text = text.lower()
            words = text.split()

            for w in words:
                str_set.add(w)
        

        for idx, word in enumerate(sorted(str_set)):
            self.word_to_id[word] = idx + 4
            self.id_to_word[idx + 4] = word

        self.vocab_size = len(str_set) + 4

    def encode(self, text: str) -> List[int]:
        words = text.split()
        res = []
        for w in words:
            w = w.lower()
            res.append(self.word_to_id.get(w, 1))
        return res
    
    def decode(self, ids: List[int]) -> str:
        res_str = ""
        for id in ids: 
            res_str += self.id_to_word.get(id, self.unk_token) + " "
        return res_str[:-1]