cityscapesC_type = "CityscapesDataset"
cityscapesC_root = "data/"
cityscapesC_crop_size = (512, 512)


cityscapesC_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
# blur: motion, defocus, glass, gaussian
val_cityscapesC_blur_motion = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="blur/motion",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_blur_defocus = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="blur/defocus",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_blur_glass = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="blur/glass",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_blur_gauss = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="blur/gaussian",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
# noise: gauss, impul, shot, speck
val_cityscapesC_noise_gauss = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="noise/gaussian",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_noise_impulse = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="noise/impulse",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_noise_shot = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="noise/shot",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_noise_speckle = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="noise/speckle",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
# digital: bright, contrast, saturate, jpeg
val_cityscapesC_digital_bright = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="digital/bright",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_digital_contrast = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="digital/contrast",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_digital_saturate = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="digital/saturate",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_digital_jpeg = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="digital/jpeg",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
# weather: snow, spatter, fog, frost
val_cityscapesC_weather_snow = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="weather/snow",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_weather_spatter = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="weather/spatter",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_weather_fog = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="weather/fog",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)
val_cityscapesC_weather_frost = dict(
    type=cityscapesC_type,
    data_root=cityscapesC_root,
    data_prefix=dict(
        img_path="weather/frost",
        seg_map_path="gt",
    ),
    img_suffix="_leftImg8bit.png",
    seg_map_suffix="_gtFine_labelTrainIds.png",
    pipeline=cityscapesC_val_pipeline,
)