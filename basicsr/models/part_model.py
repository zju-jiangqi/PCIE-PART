import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .image_restoration_model import ImageRestorationModel

@MODEL_REGISTRY.register()
class PART_Model(ImageRestorationModel):

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'psf' in data:
            self.psf = data['psf'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # self.output = self.net_g(self.lq)
        preds = self.net_g(self.lq,self.psf)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            # for pred, gt in zip(preds, gts):
            #     l_pix += self.cri_pix(pred, gt)
            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j, :, :, :],self.psf[i:j, :, :, :])
                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()