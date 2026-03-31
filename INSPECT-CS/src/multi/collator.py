import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Any

class Collator(object):
    """
    Collate function for the DataLoader
    Efficiently batches and pads sequences for model training
    """
    def __init__(self, modalities: List[str]):
        self.modalities = modalities

    def __call__(self, batch: List[Tuple[Any, Any, torch.Tensor, torch.Tensor, Any, Any]]):
        # Unpack batch (x1, x2, label, mask, pid, study)
        if len(self.modalities) == 3:
            x1, x2, x3, labels, masks, studies, impressions = zip(*batch)
        else:
            x1, x2, labels, masks, studies, impressions = zip(*batch)

        # ----------------------------
        # Handle first modality (x1)
        # ----------------------------
        if self.modalities[0] == "image":
            x1_ = torch.stack(list(x1))
            mask1_ = torch.stack(list(masks))  # mask is related to the image
        elif self.modalities[0] == "report":
            max_length = max(len(sequence) for report in x1 for sequence in report)
            processed_reports = []
            for report in x1:
                sequences = [torch.tensor(sequence) for sequence in report]
                padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
                if padded_sequence.size(1) < max_length:
                    padded_sequence = torch.nn.functional.pad(
                        padded_sequence, 
                        (0, 0, 0, max_length - padded_sequence.size(1))
                    )
                processed_reports.append(padded_sequence)
            x1_ = pad_sequence(processed_reports, batch_first=True, padding_value=0)
            token_sums = x1_.sum(dim=-1)
            mask1_ = (token_sums.sum(dim=-1) != 0).int()
        elif self.modalities[0] == "ehr":
            x1_ = torch.stack(list(x1))
            mask1_ = None

        # ----------------------------
        # Handle second modality (x2)
        # ----------------------------
        if self.modalities[1] == "image":
            x2_ = torch.stack(list(x2))
            mask2_ = torch.stack(list(masks))  # if x2 is image, use mask
        elif self.modalities[1] == "report":
            max_length = max(len(sequence) for report in x2 for sequence in report)
            processed_reports = []
            for report in x2:
                sequences = [torch.tensor(sequence) for sequence in report]
                padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
                if padded_sequence.size(1) < max_length:
                    padded_sequence = torch.nn.functional.pad(
                        padded_sequence, 
                        (0, 0, 0, max_length - padded_sequence.size(1))
                    )
                processed_reports.append(padded_sequence)
            x2_ = pad_sequence(processed_reports, batch_first=True, padding_value=0)
            token_sums = x2_.sum(dim=-1)
            mask2_ = (token_sums.sum(dim=-1) != 0).int()
        elif self.modalities[1] == "ehr":
            x2_ = torch.stack(list(x2))
            mask2_ = None
        
        # ----------------------------
        # Handle third modality (x3) if present
        # ----------------------------
        if len(self.modalities) == 3:
            if self.modalities[2] == "image":
                x3_ = torch.stack(list(x3))
                mask3_ = torch.stack(list(masks))  # if x3 is image, use mask
            elif self.modalities[2] == "report":
                max_length = max(len(sequence) for report in x3 for sequence in report)
                processed_reports = []
                for report in x3:
                    sequences = [torch.tensor(sequence) for sequence in report]
                    padded_sequence = pad_sequence(sequences, batch_first=True, padding_value=0)
                    if padded_sequence.size(1) < max_length:
                        padded_sequence = torch.nn.functional.pad(
                            padded_sequence, 
                            (0, 0, 0, max_length - padded_sequence.size(1))
                        )
                    processed_reports.append(padded_sequence)
                x3_ = pad_sequence(processed_reports, batch_first=True, padding_value=0)
                token_sums = x3_.sum(dim=-1)
                mask3_ = (token_sums.sum(dim=-1) != 0).int()
            elif self.modalities[2] == "ehr":
                x3_ = torch.stack(list(x3))
                mask3_ = None

            # Return also the third modality
            return (x1_, x2_, x3_, torch.stack(list(labels)), (mask1_, mask2_, mask3_), studies, impressions)

        # ----------------------------
        # Label
        # ----------------------------
        label = torch.stack(list(labels))

        # ----------------------------
        # Return
        # ----------------------------
        return (x1_, x2_, label, (mask1_, mask2_), studies, impressions)