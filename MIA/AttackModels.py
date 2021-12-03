from MIA.utils import *
from MIA.ShadowModels import *


class ConfidenceVector():
    def __init__(self, shadowmodel, topx=-1):
        self.shadowdata = shadowmodel.data
        #
        self.n_classes = int(max(torch.max(self.shadowdata.target_in).cpu().numpy(),torch.max(self.shadowdata.target_out).cpu().numpy())+1)
        self.topx = topx

    def train(self):
        if self.topx==-1:
            for i in range(self.n_classes):
                train_x=torch.cat(self.shadowdata.data_in[self.shadowdata.target_in==i:],self.shadowdata.data_out[self.shadowdata.target_out==i:])
        else:
            pass
