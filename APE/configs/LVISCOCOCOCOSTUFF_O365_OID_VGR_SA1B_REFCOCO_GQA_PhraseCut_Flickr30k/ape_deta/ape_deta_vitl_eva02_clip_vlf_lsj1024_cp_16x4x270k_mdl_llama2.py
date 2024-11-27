import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.solver import WarmupParamScheduler
from detrex.modeling.neck import ChannelMapper
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ape.data.detection_utils import get_fed_loss_cls_weights
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
)
from ape.modeling.text import Llama2

from ...common.backbone.vitl_eva02_clip import backbone
from ...common.data.lviscocococostuff_o365_oid_vgr_sa1b_refcoco_group_by_image_gqa_phrasecut_flickr30k_panoptic_lsj1024_cp_mdl import (
    dataloader,
)
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_24ep import (
    model,
    optimizer,
    train,
)

model.model_vision.backbone = backbone

train.init_checkpoint = (
    "models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14to16_s6B.pt?matching_heuristics=True"
)

model.model_language = L(Llama2)(
    pretrained_model_name_or_path="models/meta-llama/Llama-2-7b-hf/",
    dtype="float32",
    vision_port="decoder",
    eval_only=True,
    load_in_4bit=True,
    load_in_8bit=False,
)
model.model_vision.embed_dim_language = 4096
model.model_vision.text_feature_reduce_type = "average"

model.model_vision.neck = L(ChannelMapper)(
    input_shapes={
        "p2": ShapeSpec(channels=256),
        "p3": ShapeSpec(channels=256),
        "p4": ShapeSpec(channels=256),
        "p5": ShapeSpec(channels=256),
        "p6": ShapeSpec(channels=256),
    },
    in_features=["p2", "p3", "p4", "p5", "p6"],
    out_channels=256,
    num_outs=5,
    kernel_size=1,
    norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
)

model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

model.model_vision.transformer.encoder.num_layers = 6
model.model_vision.transformer.decoder.num_layers = 6
model.model_vision.transformer.encoder.embed_dim = 256
model.model_vision.transformer.decoder.embed_dim = 256
model.model_vision.embed_dim = 256
model.model_vision.backbone.out_channels = 256

model.model_vision.update(
    _target_=DeformableDETRSegmVL,
)
model.model_vision.transformer.update(
    _target_=DeformableDetrTransformerVL,
)
model.model_vision.transformer.encoder.update(
    _target_=DeformableDetrTransformerEncoderVL,
)
model.model_vision.transformer.decoder.update(
    _target_=DeformableDetrTransformerDecoderVL,
)

model.model_vision.transformer.encoder.vl_layer = L(VisionLanguageFusion)(
    v_dim="${....embed_dim}",
    l_dim="${....embed_dim_language}",
    embed_dim=2048,
    num_heads=8,
    dropout=0.1,
    drop_path=0.0,
    init_values=1.0 / 6,
    stable_softmax_2d=True,
    clamp_min_for_underflow=True,
    clamp_max_for_overflow=True,
    use_checkpoint=True,
    use_attention_mask_v=True,
)
model.model_vision.transformer.encoder.use_act_checkpoint = True
model.model_vision.transformer.decoder.use_act_checkpoint = True

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True
model.model_vision.name_prompt_fusion_type = "zero"

model.model_vision.num_classes = 1256
model.model_vision.select_box_nums_for_evaluation = 300

criterion = model.model_vision.criterion[0]
del criterion.use_fed_loss
del criterion.get_fed_loss_cls_weights
del criterion.fed_loss_num_classes
model.model_vision.criterion = [criterion for _ in range(10)]
for criterion, num_classes in zip(
    model.model_vision.criterion, [1256, 365, 601, 256, 1, 256, 256, 256, 256, 256]
):
    criterion.num_classes = num_classes

model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train[0].dataset.names, 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50
model.model_vision.criterion[0].fed_loss_pad_type = "cat"

model.model_vision.criterion[2].use_fed_loss = True
model.model_vision.criterion[2].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train[2].dataset.names, 0.5
)
model.model_vision.criterion[2].fed_loss_num_classes = 50
model.model_vision.criterion[2].fed_loss_pad_type = "cat"

model.model_vision.criterion[3].weight_dict["loss_class_enc"] = 0.0
for k, v in model.model_vision.criterion[3].weight_dict.items():
    if "_enc" in k:
        model.model_vision.criterion[3].weight_dict.update({k: 0.0})
    if "_bbox" in k or "_giou" in k or "_dice" in k or "_mask" in k:
        model.model_vision.criterion[3].weight_dict.update({k: 0.0})

for k, v in model.model_vision.criterion[4].weight_dict.items():
    if "_class" in k and "_enc" not in k:
        model.model_vision.criterion[4].weight_dict.update({k: 0.0})

model.model_vision.criterion[5].weight_dict["loss_class_enc"] = 0.0

model.model_vision.criterion[6].weight_dict["loss_class_enc"] = 0.0
for k, v in model.model_vision.criterion[6].weight_dict.items():
    if "_enc" in k:
        model.model_vision.criterion[6].weight_dict.update({k: 0.0})
    if "_bbox" in k or "_giou" in k or "_dice" in k or "_mask" in k:
        model.model_vision.criterion[6].weight_dict.update({k: 0.0})

model.model_vision.criterion[7].weight_dict["loss_class_enc"] = 0.0
for k, v in model.model_vision.criterion[7].weight_dict.items():
    if "_enc" in k:
        model.model_vision.criterion[7].weight_dict.update({k: 0.0})
    if "_bbox" in k or "_giou" in k or "_dice" in k or "_mask" in k:
        model.model_vision.criterion[7].weight_dict.update({k: 0.0})

model.model_vision.criterion[8].weight_dict["loss_class_enc"] = 0.0
for k, v in model.model_vision.criterion[8].weight_dict.items():
    if "_enc" in k:
        model.model_vision.criterion[8].weight_dict.update({k: 0.0})
    if "_bbox" in k or "_giou" in k or "_dice" in k or "_mask" in k:
        model.model_vision.criterion[8].weight_dict.update({k: 0.0})

model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = 0.9
model.model_vision.transformer.proposal_ambiguous = 1

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

train.max_iter = 270000
train.eval_period = 270000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[225000],
        num_updates=270000,
    ),
    warmup_length=2000 / 270000,
    warmup_method="linear",
    warmup_factor=0.001,
)

for i in range(len(dataloader.train)):
    dataloader.train[i].mapper.max_num_phrase = 128
    dataloader.train[i].mapper.nms_thresh_phrase = 0.6
    dataloader.train[i].total_batch_size = 16
    dataloader.train[i].total_batch_size_list = [16]
    dataloader.train[i].num_workers = 2

train.iter_size = 4
train.dataset_ratio = [1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]

model.model_vision.dataset_prompts = [
    "name",
    "name",
    "name",
    "phrase",
    "name",
    "phrase",
    "phrase",
    "phrase",
    "phrase",
    "expression",
]
model.model_vision.dataset_names = [
    "lvis+stuffonly",
    "objects365",
    "openimages",
    "vgregion",
    "sa1b",
    "refcoco-mixed_group-by-image",
    "gqa",
    "phrasecut",
    "flickr30k",
    "refcoco",
]
model.model_vision.dataset_metas = [xx for x in dataloader.train for xx in x.dataset.names] + [
    "refcoco-mixed"
]

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120
