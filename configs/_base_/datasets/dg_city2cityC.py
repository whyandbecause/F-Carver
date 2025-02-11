_base_ = [
    "./citysC_512x512.py",
    "./cityscapes_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_cityscapes}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
             {{_base_.val_cityscapesC_blur_motion}},
             {{_base_.val_cityscapesC_blur_defocus}},
             {{_base_.val_cityscapesC_blur_glass}},
             {{_base_.val_cityscapesC_blur_gauss}},
            {{_base_.val_cityscapesC_noise_gauss}},
            {{_base_.val_cityscapesC_noise_impulse}},
            {{_base_.val_cityscapesC_noise_shot}},
            {{_base_.val_cityscapesC_noise_speckle}},
             {{_base_.val_cityscapesC_digital_bright}},
             {{_base_.val_cityscapesC_digital_saturate}},
             {{_base_.val_cityscapesC_digital_jpeg}},
             {{_base_.val_cityscapesC_digital_contrast}},
             {{_base_.val_cityscapesC_weather_snow}},
             {{_base_.val_cityscapesC_weather_spatter}},
             {{_base_.val_cityscapesC_weather_fog}},
             {{_base_.val_cityscapesC_weather_frost}},
        ],
    ),
)
test_dataloader = val_dataloader
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset={{_base_.val_cityscapes}},
# )
val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    # dataset_keys=["blur/motion/", "blur/defocus/", "blur/glass/", "blur/gaussian/"],
    dataset_keys=[
                     "blur/motion", "blur/defoc", "blur/glass", "blur/gauss", 
                  "noise/gauss", "noise/impulse", "noise/shot", "noise/speckle",
                  "digital/bright", "digital/contrast", "digital/saturate", "digital/jpeg",
                   "weather/snow", "weather/spatter", "weather/fog", "weather/frost"
                  ],
    # mean_used_keys=["motion/", "defoc/", "glass/", "gauss/"],
)
test_evaluator = val_evaluator
# test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
