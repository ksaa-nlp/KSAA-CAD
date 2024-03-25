from torch.utils.data import Dataset 
from utils import read_json

class ARDDataset(Dataset):
    def __init__(self, path, is_test=False) -> None:
        super().__init__()
        self.is_test = is_test
        self.data = read_json(path)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.is_test:
            return sample["id"], sample["word"], sample["gloss"],
        else:
            return sample["id"], sample["word"], sample["gloss"], sample["electra"], sample["bertseg"], sample['bertmsa']
    
    def __len__(self):
        return len(self.data)
