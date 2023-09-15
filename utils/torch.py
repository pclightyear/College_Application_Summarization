from torch.utils.data import Dataset

class BatchSentenceDataset(Dataset):
    def __init__(self, sentences):
        ## list of sentences
        self.sentences = sentences
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    
class ListDataset(Dataset):
    def __init__(self, items):
        ## list of sentences
        self.items = items
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]