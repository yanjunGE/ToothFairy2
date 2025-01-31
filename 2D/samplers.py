
import numpy as np
from torch.utils.data.sampler import Sampler
class PartialPatientSampler(Sampler):
    def __init__(self, dataset, patient_ids, num_slices=32, shuffle=True):
        self.dataset = dataset
        self.patient_to_indices = dataset.patient_to_indices
        self.patient_ids = patient_ids
        self.num_slices = num_slices
        self.shuffle = shuffle

    def __iter__(self):
        for patient_id in self.patient_ids:
            indices = self.patient_to_indices[patient_id]
            if self.shuffle:
                sampled_indices = np.random.choice(indices, size=self.num_slices, replace=False)
            else:
                sampled_indices = indices[:self.num_slices]
            # print(sampled_indices)
            yield from sampled_indices

    def __len__(self):
        return len(self.patient_ids) * self.num_slices

