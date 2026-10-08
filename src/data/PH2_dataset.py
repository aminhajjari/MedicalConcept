from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class PH2Dataset(Dataset):

    def __init__(self, csv_file, img_extension, path_to_images, img_prefix=""):
        self.data = pd.read_csv(csv_file)
        self.img_extension = img_extension
        self.path_to_images = path_to_images
        self.img_prefix = img_prefix
        self._base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_id = sample["images"]
        img_path = os.path.join(self.path_to_images, self.img_prefix + img_id + "." + self.img_extension)
        class_label = sample["labels"]
        derm_concepts_df = pd.read_csv(os.path.join(self._base, "data/Derm7pt/splits/dermoscopic_concepts_PH2_Derm7pt.csv"))
        derm_concepts_sample = derm_concepts_df.loc[derm_concepts_df.image_id == img_id]
        derm_concepts = np.array([derm_concepts_sample["TPN"],
                         derm_concepts_sample["APN"],
                         derm_concepts_sample["ISTR"],
                         derm_concepts_sample["RSTR"],
                         derm_concepts_sample["RDG"],
                         derm_concepts_sample["IDG"],
                         derm_concepts_sample["BWV"],
                         derm_concepts_sample["RS"]])
        return {"img_id": img_id, "img_path": img_path, "class_label": class_label, "derm_concepts": derm_concepts.squeeze()}
