data_name = "schs-lung"
batch_size = 32

postnet_param = dict(
    # Dataset
    dataset_name=data_name,
    # Model
    architecture='linear',
    input_dims=[19],
    output_dim=2,
    hidden_dims=[8, 8, 8],
    kernel_dim=None,
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
    lr=1e-3,
    loss='UCE',
    training_mode='joint',
    regr=1e-5,
    timing=False
)