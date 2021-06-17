from ia.int_models.rts_densenet import RTSaliencyModel, resnet50encoder
from ia_utils.data_utils import imagenet_normalize
from torchvision.models import densenet169


class RTSDensenet169(object):

    def __init__(self, ckpt_dir, cuda, blackbox_model=None, pre_fn=None):
        self.saliency = RTSaliencyModel(resnet50encoder(pretrained=True), 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=True)
        self.saliency.minimialistic_restore(ckpt_dir)
        self.saliency.train(False)
        if cuda:
            self.saliency.cuda()

        if blackbox_model is None:
            blackbox_model = densenet169(pretrained=True)
            self.blackbox_model = blackbox_model
            self.blackbox_model.train(False)
            if cuda:
                self.blackbox_model.cuda()
        else:
            self.blackbox_model = blackbox_model
        self.pre_fn = imagenet_normalize if pre_fn is None else pre_fn

    def saliency_fn(self, x, y, model_confidence=6, return_classification_logits=False):
        masks, _, cls_logits = self.saliency(imagenet_normalize(x), y,
                                             model_confidence=model_confidence)
        # sal_map = F.upsample(masks, (x.size(2), x.size(3)), mode='bilinear')
        if not return_classification_logits:
            return masks
        return masks, cls_logits

    def logits_fn(self, x):
        logits = self.saliency.encoder(imagenet_normalize(x))[-1]
        return logits

    def blackbox_logits_fn(self, x):
        return self.blackbox_model(self.pre_fn(x))
