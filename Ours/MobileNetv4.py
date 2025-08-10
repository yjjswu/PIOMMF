import timm
model=timm.create_model(
    'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k',
    pretrained=False,
)