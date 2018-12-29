class Config:
    batch_size = 1
    img_channel = 1
    conv_channel_base = 64
    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    save_per_epoch=1
    
    trained_models_path = "/data/dung/sargan/trained_models"
    output_path = "/data/dung/sargan/outputs"
    model_path_train = ""
#     trained_models_path = "/data/dung/sargan/radarconf19_v3/trained_models"
#     output_path = "/data/dung/sargan/radarconf19_v3/outputs"
#     model_path_train = ""