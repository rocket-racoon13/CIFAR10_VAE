import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


class Predictor:
    def __init__(
        self,
        args,
        eval_ds,
        model,
        device
    ):
        self.args = args
        self.eval_ds = eval_ds
        self.model = model
        self.device = device
        
    def predict(self):
        predictions = []
        
        eval_dataloader = DataLoader(
            dataset=self.eval_ds,
            batch_size=len(self.eval_ds),
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch.to(self.device)
                y_pred = self.model(batch)
                _, batch_pred_labels = torch.max(y_pred, dim=1)
                batch_pred_labels = batch_pred_labels.detach().cpu()
                predictions.extend(batch_pred_labels.tolist())
        
        predictions = list(map(str, predictions))
        out_dir = os.path.join(self.args.eval_out_dir, "predictions.txt")
        with open(out_dir, "w", encoding="utf-8-sig") as f_out:
            f_out.write("\n".join(predictions))