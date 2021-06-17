import numpy as np
import pandas as pd


NUM_FOLDS = 15

BIT_DEPTHS = [1, 2, 3, 4, 5]
BIT_DEPTHS_STR = [str(i) for i in BIT_DEPTHS]
BIT_DEPTHS = [None] + BIT_DEPTHS
BIT_DEPTHS_STR = ['none'] + BIT_DEPTHS_STR

MEDIAN_SMOOTHING = [None, (2, 2), (3, 3)]
MEDIAN_SMOOTHING_STR = ['none', '2 * 2', '3 * 3']
NONLOCAL_MEAN = [None, (11, 3, 2), (11, 3, 4), (13, 3, 2), (13, 3, 4)]
NONLOCAL_MEAN_STR = ['none', '11-3-2', '11-3-4', '13-3-2', '13-3-4']


RESNET50_MODEL_PATH = '/home/xinyang/Data/intattack/detector_resnet50_model.pkl'
RESNET50_MODEL_V2_PATH = '/home/xinyang/Data/intattack/detector_resnet50_model_v2.pkl'
DENSENET169_MODEL_PATH = '/home/xinyang/Data/intattack/detector_densenet169_model.pkl'


RECORDER_BENIGN = 0
RECORDER_FAILED_ADV = 1
RECORDER_ADV = 2
RECORDER_STR = ['benign', 'failed adv', 'adv']

PRED_AS_BENIGN = 0
PRED_AS_ADV = 1
PRED_AS_STR = ['benign', 'adv']


LID_DATA_PATH = 'fold_%d.npz'
LID_ADV_DATA_PATH = 'adv_fold_%d.npz'
LID_BENIGN_DATA_DIR = '/home/xinyang/Data/intattack/lid_val_data_resnet50'
# LID_ACID_RTS_DATA_PATH = 'acid_rts_fold_%d.npz'


LID_MODEL_REGULAR_PGD = '/home/xinyang/Data/intattack/lid_model_reg_pgd.pkl'
LID_MODEL_ACID_RTS = '/home/xinyang/Data/intattack/lid_model_acid_rts.pkl'


class Recorder():
    def __init__(self):
        self.image_types = []
        self.preds = []
        self.descriptions = []

    def append(self, image_type, pred, description):
        self.image_types.append(image_type)
        self.preds.append(pred)
        self.descriptions.append(description)

    def output(self):
        d = {
            "image_type": np.asarray(self.image_types, np.int64),
            "pred": np.asarray(self.preds, np.int64),
            "description": self.descriptions
        }
        df = pd.DataFrame(data=d)
        return df

    def produce_stats(self):
        df = self.output()
        df['image_type'] = df['image_type'].map(lambda x: RECORDER_STR[x])
        df['pred'] = df['pred'].map(lambda x: PRED_AS_STR[x])

        return df.pivot_table(index=['description', 'image_type'], columns=['pred'], aggfunc=len,
                              fill_value=0)
