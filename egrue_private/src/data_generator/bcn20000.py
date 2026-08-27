from os.path import join
from torchvision import transforms

from src.configs.isic_config import IMG_SIZE, num_classes, target_col
from src.data_processing.dataset import ImageDataset

def load_bcn20000_data_dict(fp_processed_data_folder):
    transform = transforms.Compose([
        transforms.Resize(size=IMG_SIZE),
        transforms.ToTensor(),
    ])
    fp_csv_file_in = join(fp_processed_data_folder, "metadata_in.csv")
    fp_csv_file_out = join(fp_processed_data_folder, "metadata_out.csv")
    fp_img_dir = join(fp_processed_data_folder, "images")
    test_ds_in = ImageDataset(
        fp_csv_file=fp_csv_file_in, fp_img_dir=fp_img_dir,
        img_fn_col="image_fn", target_col=target_col, transform=transform
    )
    data_dict_in =  {
        "target_col": target_col,
        "data_name": "bcn20000",
        "num_classes": num_classes+1,
        "classes": ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC', "Scar"],
        "test_df": test_ds_in,
    }
    test_ds_out = ImageDataset(
        fp_csv_file=fp_csv_file_out, fp_img_dir=fp_img_dir,
        img_fn_col="image_fn", target_col=target_col, transform=transform
    )
    data_dict_out =  {
        "target_col": target_col,
        "data_name": "bcn20000",
        "num_classes": num_classes+1,
        "classes": ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC', "Scar"],
        "test_df": test_ds_out,
    }
    return data_dict_in, data_dict_out