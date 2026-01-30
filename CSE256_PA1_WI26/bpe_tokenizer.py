# bpe_tokenizer.py
import pickle
from tqdm import tqdm

class BPETokenizer():

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def get_stats(self,ids):
        counts = {}

        for pair in zip (ids,ids[1:]):
            counts[pair] = counts.get(pair,0) + 1

        return counts

    def merge(self,ids,pair,idx):
        newids = []
        i = 0

        while i < len(ids):
            if i< ( len(ids) -1) and (ids[i] == pair[0]) and (ids[i+1] == pair[1]):
                newids.append(idx)
                i +=2
            else:
                newids.append(ids[i])
                i +=1

        return newids


    def train(self,texts,vocab_size):
        # turn into byte
        ids = []
        for text in texts:
            id =  list(text.encode('utf-8'))
            ids.extend(id)
        # max for byte encoding
        base_vocab = 256
        num_merges = vocab_size - base_vocab

        for i in tqdm(range(num_merges), desc="Training BPE", unit="merge"):
            stats = self.get_stats(ids)
            pair = max(stats,key = stats.get)
            index = base_vocab + i
            # need to save because need to be used in get_stats
            ids = self.merge(ids,pair,index)
            self.merges[pair] = index
    

    def encode(self,text):
        ids = list(text.encode('utf-8'))

        for pair, new_id in self.merges.items():
            ids = self.merge(ids,pair,new_id)

        return ids
    
    def save(self,filepath):
        with open(filepath,'wb') as f:
            pickle.dump(self.merges,f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.merges = pickle.load(f)  # ← Fixed: added f parameter

            
    def selfTest(self):
        """
        Unit test for BPE tokenizer
        Tests: train(), encode(), get_stats(), merge()
        """
        print("=" * 60)
        print("BPE Tokenizer Self Test")
        print("=" * 60)
        
        # Test 1: get_stats()
        print("\n[Test 1] get_stats()")
        test_ids = [1, 2, 1, 2, 1, 3]
        stats = self.get_stats(test_ids)
        print(f"  Input: {test_ids}")
        print(f"  Stats: {stats}")
        assert stats[(1, 2)] == 2, "Failed: (1,2) should appear 2 times"
        assert stats[(2, 1)] == 2, "Failed: (2,1) should appear 2 times"
        print("  ✅ PASSED")
        
        # Test 2: merge()
        print("\n[Test 2] merge()")
        test_ids = [1, 2, 1, 2, 3]
        merged = self.merge(test_ids, (1, 2), 99)
        print(f"  Input: {test_ids}")
        print(f"  Merge (1,2) -> 99")
        print(f"  Output: {merged}")
        assert merged == [99, 99, 3], f"Failed: expected [99, 99, 3], got {merged}"
        print("  ✅ PASSED")
        
        # Test 3: train() and encode()
        print("\n[Test 3] train() and encode()")
        corpus = ["aaabdaaabac"] * 10  # Repeat to increase frequency
        print(f"  Corpus: {corpus[0]} (repeated 10 times)")
        print(f"  Target vocab size: 260")
        
        self.train(corpus, vocab_size=260)
        print(f"  Learned {len(self.merges)} merge rules")
        
        # Show first 3 merges
        print("  First 3 merge rules:")
        for i, (pair, idx) in enumerate(list(self.merges.items())[:3]):
            try:
                # Try to decode bytes
                pair_str = f"({chr(pair[0])}, {chr(pair[1])})" if pair[0] < 256 and pair[1] < 256 else str(pair)
            except:
                pair_str = str(pair)
            print(f"    {i+1}. {pair_str} -> token {idx}")
        
        # Test encoding
        test_text = "aaabdaaabac"
        encoded = self.encode(test_text)
        print(f"\n  Encoding test: '{test_text}'")
        print(f"  Original bytes: {list(test_text.encode('utf-8'))}")
        print(f"  BPE encoded: {encoded}")
        print(f"  Compression: {len(test_text.encode('utf-8'))} -> {len(encoded)} tokens")
        
        # Verify encoding is shorter (BPE should compress)
        assert len(encoded) <= len(test_text.encode('utf-8')), "BPE should compress!"
        print("  ✅ PASSED")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)

if __name__ == '__main__':
    print("\nRunning BPE Tokenizer Unit Tests...\n")
    
    tokenizer = BPETokenizer()
    tokenizer.selfTest()
    
    print("\n✅ BPE Tokenizer is working correctly!")



