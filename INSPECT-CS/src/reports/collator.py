import torch
from torch.nn.utils.rnn import pad_sequence

class Collator(object):
    """
        Collate function for the DataLoader
    """
    def __init__(self):
        """
        Initialize the collator with modalities.
        """

    def __call__(self, batch):
        # Collate Inputs and Labels
        max_length = max(len(sample) for input, label, pid in batch for sample in input)
        inputs = []
        for input, _, _ in batch:
            sequences = [torch.tensor(sample, dtype=torch.float32) for sample in input]
            padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
            padded_sequence = torch.nn.functional.pad(padded_sequence, (0, 0, max_length - padded_sequence.size(1), 0))
            inputs.append(padded_sequence)

        # Padding 
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

        labels = torch.stack([label for _, label, _ in batch])  # Assuming labels are tensors and need to be stacked

        pids = [pid for _, _, pid in batch]  # Assuming pids are strings or integers

        return inputs, labels, pids