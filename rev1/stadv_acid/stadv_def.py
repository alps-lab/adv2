import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StadvTVLoss(nn.Module):

    def forward(self, flows):
        padded_flows = F.pad(flows, (1, 1, 1, 1), mode='replicate')
        height, width = flows.size(2), flows.size(3)
        n = float(np.sqrt(height * width))
        shifted_flows = [
            padded_flows[:, :, 2:, 2:],
            padded_flows[:, :, 2:, :-2],
            padded_flows[:, :, :-2, 2:],
            padded_flows[:, :, :-2, :-2]
        ]

        diffs = [(flows[:, 1] - shifted_flow[:, 1]) ** 2 + (flows[:, 0] - shifted_flow[:, 0]) ** 2
                 for shifted_flow in shifted_flows]
        loss = torch.stack(diffs).sum(2, keepdim=True).sum(3, keepdim=True).sum(0, keepdim=True).view(-1)
        loss = torch.sqrt(loss)
        return loss / n


class StadvFlowLoss(nn.Module):

    def forward(self,flows, epsilon=1e-8):
        padded_flows = F.pad(flows, (1, 1, 1, 1), mode='replicate')
        shifted_flows = [
            padded_flows[:, :, 2:, 2:],
            padded_flows[:, :, 2:, :-2],
            padded_flows[:, :, :-2, 2:],
            padded_flows[:, :, :-2, :-2]
        ]

        diffs = [torch.sqrt((flows[:, 1] - shifted_flow[:, 1]) ** 2 +
                            (flows[:, 0] - shifted_flow[:, 0]) ** 2 +
                            epsilon) for shifted_flow in shifted_flows
                 ]
        # shape: (4, n, h - 1, w - 1) => (n, )
        loss = torch.stack(diffs).sum(2, keepdim=True).sum(3, keepdim=True).sum(0, keepdim=True).view(-1)
        return loss


class StadvFlow(nn.Module):

    def forward(self, images, flows):
        batch_size, n_channels, height, width = images.shape
        basegrid = torch.stack(torch.meshgrid([torch.arange(height, device=images.device),
                                               torch.arange(width, device=images.device)]))
        batched_basegrid = basegrid.expand(batch_size, -1, -1, -1)

        sampling_grid = batched_basegrid.float() + flows
        sampling_grid_x = torch.clamp(sampling_grid[:, 1], 0., float(width) - 1)
        sampling_grid_y = torch.clamp(sampling_grid[:, 0], 0., float(height) - 1)

        x0 = sampling_grid_x.floor().long()
        x1 = x0 + 1
        y0 = sampling_grid_y.floor().long()
        y1 = y0 + 1

        x0.clamp_(0, width - 2)
        x1.clamp_(0, width - 1)
        y0.clamp_(0, height - 2)
        y1.clamp_(0, height - 1)

        b = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, height, width)

        Ia = images[b, :, y0, x0].permute(0, 3, 1, 2)
        Ib = images[b, :, y1, x0].permute(0, 3, 1, 2)
        Ic = images[b, :, y0, x1].permute(0, 3, 1, 2)
        Id = images[b, :, y1, x1].permute(0, 3, 1, 2)

        x0 = x0.float()
        x1 = x1.float()
        y0 = y0.float()
        y1 = y1.float()

        wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
        wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
        wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
        wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

        wa = wa.unsqueeze(1)
        wb = wb.unsqueeze(1)
        wc = wc.unsqueeze(1)
        wd = wd.unsqueeze(1)

        perturbed_image = torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id]).sum(0)
        return perturbed_image


if __name__ == '__main__':
    h = np.load('/home/xinyang/Data/intattack/fixup1/resnet_data/fold_1.npz')
    img = h['img_x']
    images = img[:4]
    flows = torch.zeros(4, 2, 224, 224, device='cuda')
    flows[:, 0] = 1
    images = torch.tensor(images, device='cuda')

    Flow = StadvFlow()
    perturbed_images = Flow(images, flows)

    FlowLoss = StadvFlowLoss()
    loss = FlowLoss(flows)
    print(loss)
