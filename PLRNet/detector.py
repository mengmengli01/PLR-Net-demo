import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from PLRNet.backbones import build_backbone
from PLRNet.utils.polygon import generate_polygon
from PLRNet.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops


class RSCSEModule(nn.Module):
    def __init__(self, in_channels=32, reduction=4):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x, x1):
        return x + x * self.cSE(x1) + x * self.sSE(x1)


class RAMAttention(nn.Module):

    def __init__(self, dim_in=2, dim_hid=16, dim_out=32, reduction=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim_out, dim_out // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out // reduction, dim_out, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.layer(x)
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x1 * y.expand_as(x1)



class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.junc_loss = nn.CrossEntropyLoss()

        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        # self.test_inria = 'inria' not in cfg.DATASETS.TEST[0]
        if not test:
    
            from hisup.encoder import Encoder
            self.encoder = Encoder(cfg)

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        # self.a2m_att = ECA(dim_in)
        # self.a2j_att = ECA(dim_in)
        self.a2m_att = RSCSEModule(dim_in, 4)
        self.a2j_att = RSCSEModule(dim_in, 4)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        self.refuse_conv = RAMAttention(2, dim_in // 2, dim_in, 4)
        # self.final_conv = self._make_conv(dim_in*2, dim_in, 2)
        self.final_conv = self._make_conv(dim_in, dim_in // 2, 2)

        self.train_step = 0

    def forward(self, images, annotations=None):
        return self.forward_test(images, annotations=annotations)

    def forward_test(self, images, annotations=None):
        device = images.device
        outputs, features = self.backbone(images)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        # mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        # jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)
        mask_att_feature = self.a2m_att(mask_feature, mask_feature + afm_feature)
        jloc_att_feature = self.a2j_att(jloc_feature, jloc_feature + afm_feature)

        # mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        # jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        mask_pred = self.mask_predictor(mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        # remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))
        remask_pred = self.final_conv(features + afm_conv)

        joff_pred = outputs[:, :].sigmoid() - 0.5

        mask_pred = mask_pred.softmax(1)[:, 1:]

        # mask_pred = mask_pred.softmax(1)

        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]

        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]

        # remask_pred = mask_pred
        remask_pred = remask_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):

            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))

            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])

            juncs_pred[:, 0] *= scale_x
            juncs_pred[:, 1] *= scale_y

            if not self.test_inria:
                polys, scores = [], []


                props = regionprops(label(mask_pred_per_im > 0.5))
                for prop in props:


                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
                                                                                    juncs_pred, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue

                    polys.append(poly) 
                    scores.append(score)  
                batch_scores.append(scores)  
                batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)  
            batch_juncs.append(juncs_pred)

        extra_info = {}

        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info

        # return output, mask

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer


    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
            nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, dim_out, kernel_size=1),
        )
        return layer



    model = BuildingDetector(cfg, test=True)
    if pretrained:
        url = PRETRAINED[dataset]
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
        state_dict = {k[7:]: v for k, v in state_dict['model'].items() if k[0:7] == 'module.'}
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
     return model
