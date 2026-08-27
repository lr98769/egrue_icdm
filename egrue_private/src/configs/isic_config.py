data_name = "isic"
num_classes = 7
target_col = "class"
IMG_SIZE = 224
batch_size = 64
eval_batch_size = 64*4


seed_interval_size=100
ensemble_size=5

# Melanoma: MEL
# Melanocytic nevus: NV
# Basal cell carcinoma: BCC
# Actinic keratosis / Bowen’s disease (intraepithelial carcinoma): AKIEC
# Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis): BKL
# Dermatofibroma: DF
# Vascular lesion: VASC
# ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

postnet_param = dict(
    # Dataset
    dataset_name=data_name,
    # Model
    architecture="conv",
    input_dims=[224, 224, 3],
    output_dim=num_classes,
    hidden_dims=[64, 128, 256, 512],
    kernel_dim=5,
    latent_dim=6,
    no_density=False,
    density_type='radial_flow',
    n_density=6,
    k_lipschitz=None,
    budget_function='id',
    # Training Params
    max_epochs=500, # 500
    patience=5, # Change to be consistent with existing experiment parameters
    frequency=2,
    batch_size=batch_size, # Change to be consistent with existing experiment parameters
    lr=5e-5,
    loss='UCE',
    training_mode='joint',
    regr=1e-5,
    timing=False
)




