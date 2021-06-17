import torch

from ia.int_models.rt_saliency import RTSaliencyModel
from torchvision.models.resnet import resnet50
from ia_utils.data_utils import imagenet_normalize

from rev1.isic_acid.isic_model import get_isic_model_on_resnet50
from rev1.isic_acid.resnet_encoder_isic import resnet50encoder
from rev1.isic_acid.isic_utils import ISIC_RESNET50_CKPT_PATH

RTS_ISIC_CKPT_PATH = '/home/xinyang/ClonedCodes/pytorch-saliency/isic_phase2_resnet/'


class RTSResnet50(object):

    def __init__(self, ckpt_dir, cuda, blackbox_model=None):
        encoder = resnet50encoder(7, pretrained=True)
        # encoder.load_state_dict(torch.load(ISIC_RESNET50_CKPT_PATH)['state_dict'])
        encoder.train(False)
        self.saliency = RTSaliencyModel(encoder, 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=True, num_classes=7)
        self.saliency.minimialistic_restore(ckpt_dir)
        self.saliency.train(False)
        if cuda:
            self.saliency.cuda()

        if blackbox_model is None:
            blackbox_model = get_isic_model_on_resnet50(ckpt_path=ISIC_RESNET50_CKPT_PATH)
            self.blackbox_model = blackbox_model
            if cuda:
                self.blackbox_model.cuda()
        else:
            self.blackbox_model = blackbox_model

    def saliency_fn(self, x, y, model_confidence=0., return_classification_logits=False):
        masks, _, cls_logits = self.saliency(imagenet_normalize(x), y, model_confidence=model_confidence)
        # sal_map = F.upsample(masks, (x.size(2), x.size(3)), mode='bilinear')
        if not return_classification_logits:
            return masks
        return masks, cls_logits

    def logits_fn(self, x):
        logits = self.saliency.encoder(x)[-1]
        return logits

    def blackbox_logits_fn(self, x):
        return self.blackbox_model(x)
