from torch.utils.data import Dataset




class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch



def map_fn(x):
    x["short_answer"] = None
    x["long_answer"] = None
    return x



def update_map(x, idx, original_idxs, long_answers, short_answers):
    if idx in original_idxs:
        x["long_answer"] = long_answers[original_idxs.index(idx)]
        x["short_answer"] = (
            float(short_answers[original_idxs.index(idx)])
            if short_answers[original_idxs.index(idx)]
            else None
        )

    for key in x:
        if type(x[key]) in [list, dict]:
            x[key] = str(x[key])
    return x
