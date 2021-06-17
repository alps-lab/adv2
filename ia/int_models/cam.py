#!/usr/bin/env python
import numpy as np
import torch


class CAM(object):

    def __init__(self, model, output_keys=None):
        self.model = model
        self.model.cuda()
        self.model.train(False)
        if output_keys is not None:
            self.output_keys = list(output_keys)
        else:
            self.output_keys = ["l4", "gvp", "fc"]

    def forward(self, x, out_keys=None):
        if out_keys is None:
            out_keys = self.output_keys
        res = self.model(x, out_keys)

        rets = [res[key] for key in out_keys]
        return rets

    def __call__(self, x, y=None, out_keys=None):
        if y is None:
            with torch.no_grad():
                _, _, logits = self.forward(x, out_keys=out_keys)
                logits = logits.cpu().numpy()[0]
            true_label = int(np.argmax(logits))
            y = torch.tensor([true_label]).cuda()

        l4, gvp, logits = self.forward(x)
        wc = self.model.fc.weight[y].view(1, -1, 1, 1)
        prod = (wc * l4).sum(1)

        return logits, prod
