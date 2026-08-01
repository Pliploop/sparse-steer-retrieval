import logging
import os
import random
from hashlib import sha256

import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import TextAudioDataset
from steerable_retrieval.utils.instantiators import instantiate


def instantiate_datasets(datasets_cfg):
    datasets = []
    dataset_names = []
    for dataset_name in datasets_cfg:
        dataset = instantiate(datasets_cfg[dataset_name])
        datasets.append(dataset)
        dataset_names.append(dataset_name)
    return datasets, dataset_names


def get_song_describer_annotations(data_path = None, csv_path = None, val_split = 0.1):
    
    df = pd.read_csv(csv_path)
    
    
    df = df[['path','caption','is_valid_subset','caption_id']].rename(columns = {'path':'file_path'})
    df['file_path'] = os.path.join(data_path) + '/' + df['file_path']
    #replace .mp3 with .2min.mp3
    df['file_path'] = df['file_path'].apply(lambda x: x.replace('.mp3','.2min.mp3'))
    
    records = df.to_dict(orient = 'records')
    
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    if val_split == 0.0:
        print('No validation split')
        for record in records:
            record['split'] = 'train'
        return records
    
    train_indices, val_indices = train_test_split(range(len(records)), test_size = val_split, random_state = 42)
    
    for idx in train_indices:
        records[idx]['split'] = 'train'
        
    for idx in val_indices:
        records[idx]['split'] = 'val'
    
    return records



def get_musiccaps_annotations(data_path = None, csv_path = None, val_split = 0.1, test_split = 0.1):
        
        df = pd.read_csv(csv_path)
        df['file_path'] = data_path + '/' + df['ytid'] + '.wav'
        
        records = df.to_dict(orient = 'records')
        
        for record in records:
            record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
            
        if val_split == 0.0:
            print('No validation split')
            for record in records:
                record['split'] = 'train'
            return records
        
        # split into train, val, test
        train_indices, test_indices = train_test_split(range(len(records)), test_size = test_split + val_split, random_state = 42)
        val_indices, test_indices = train_test_split(test_indices, test_size = test_split/(test_split + val_split), random_state = 42)
        
        
        
        for idx in train_indices:
            records[idx]['split'] = 'train'
            
        for idx in val_indices:
            records[idx]['split'] = 'val'
            
        for idx in test_indices:
            records[idx]['split'] = 'test'
            
        return records
    
def get_musiccaps_truncated_annotations(data_path = None, csv_path = None, val_split = 0.1, test_split = 0.1):
    df = pd.read_csv(csv_path)
    df['file_path'] = data_path + '/' + df['ytid'] + '.wav'
    
    import random
    
    records = df.to_dict(orient = 'records')
    
    print('Truncating captions')
    
    for record in records:
        # select a random sentence from the caption
        sentences = record['caption'].split('.')
        random_sentence = random.choice(sentences)
        record['caption'] = {sha256(random_sentence.encode('utf-8')).hexdigest(): random_sentence}
        
    if val_split == 0.0:
        print('No validation split')
        for record in records:
            record['split'] = 'train'
        return records
    
    # split into train, val, test
    train_indices, test_indices = train_test_split(range(len(records)), test_size = test_split + val_split, random_state = 42)
    val_indices, test_indices = train_test_split(test_indices, test_size = test_split/(test_split + val_split), random_state = 42)
    
    for idx in train_indices:
        records[idx]['split'] = 'train'
        
    for idx in val_indices:
        records[idx]['split'] = 'val'
        
    for idx in test_indices:
        records[idx]['split'] = 'test'
        
    return records


def get_maxcaps_annotations(data_path = None, csv_path = None):
    """Read JSONL file line-by-line to avoid memory issues with large files."""

    df = pd.read_csv(csv_path)
    df['file_path'] = data_path + '/' + df['file_path']
    records = df.to_dict(orient = 'records')
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    return records

def get_yt8m_annotations(data_path = None, csv_path = None):
    df = pd.read_csv(csv_path)
    df['file_path'] = data_path + '/' + df['file_path']
    records = df.to_dict(orient = 'records')
    for record in records:
        record['caption'] = {sha256(record['caption'].encode('utf-8')).hexdigest(): record['caption']}
    return records

def get_music4all_annotations(manifest_csv=None, audio_status_ok=True):
    """Annotations for music4all from the MuQ-MuLan embedding manifest.

    ``file_path`` is set directly to the pre-extracted ``.npy`` audio-embedding path so
    the preextracted-feature loader reads it as-is. Splits (train/val/test) and captions
    are taken from the manifest. Only rows with a successfully embedded audio are kept.
    """
    df = pd.read_csv(manifest_csv)
    if audio_status_ok and 'audio_embedding_status' in df.columns:
        df = df[df['audio_embedding_status'] == 'ok']
    df = df[df['audio_embedding_path'].notna()]

    records = []
    for row in df.itertuples(index=False):
        caption = str(getattr(row, 'caption', '') or '')
        split = getattr(row, 'split', 'train')
        records.append({
            'file_path': row.audio_embedding_path,
            'caption': {sha256(caption.encode('utf-8')).hexdigest(): caption},
            'split': split if split in ('train', 'val', 'test') else 'train',
            'track_id': getattr(row, 'track_id', None),
        })
    logging.info(f"Loaded {len(records)} music4all annotations from {manifest_csv}")
    return records


def get_folder_annotations(data_path = None):
    
    # recursively get all files in the data_path directory that are audio files, and their paths
    audio_files = []
    
    for root, dirs, files in os.walk(data_path):
        audio_files += [os.path.join(root, file) for file in files if file.endswith('.wav') or file.endswith('.mp3')]
        
    records = [{'file_path': file, 'caption': '', 'split': 'train'} for file in audio_files]
    
    logging.info(f"Found {len(records)} audio files in {data_path}")


    return records
    
    

class TextAudioDataModule(LightningDataModule):
    
    def __init__(self,
    datasets,
    return_audio = True,
    return_text = True,
    concept = None, 
    target_n_samples = 96000, 
    target_sr = 48000, 
    batch_size = 32, num_workers = 0, preextracted_features = False, truncate_preextracted = 50, root_dir = None, new_dir = None,
    **kwargs):


        super().__init__()
        self.annotations = []
        dataset_names, datasets = list(datasets.keys()), list(datasets.values())




        self.datasets = datasets
        self.dataset_names = dataset_names
        for dataset in self.datasets:
            dataset.split = dataset.split

        self.return_audio = return_audio
        self.return_text = return_text
        self.concept = concept
        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preextracted_features = preextracted_features
        self.truncate_preextracted = truncate_preextracted
        
        
        self.truncate_preextracted = [self.truncate_preextracted for _ in range(len(self.datasets))]
        
        self.root_dirs = [dataset.root_dir for dataset in self.datasets]
        self.new_dirs = [dataset.new_dir for dataset in self.datasets]
        
        self.train_annotations = []
        self.val_annotations = []
        self.test_annotations = []
        self.val_dataset_names = []
        self.test_dataset_names = []
        self.val_dataset_indices = []  # Track original dataset indices for val
        self.test_dataset_indices = []  # Track original dataset indices for test
        
        
        for i, dataset_ in enumerate(self.datasets):
            self.train_annotations.extend([annot for annot in dataset_.annotations if annot['split'] == 'train'])
            val_annots = [annot for annot in dataset_.annotations if annot['split'] == 'val']
            test_annots = [annot for annot in dataset_.annotations if annot['split'] == 'test']
            if val_annots:
                self.val_annotations.append(val_annots)
                self.val_dataset_names.append(self.dataset_names[i])
                self.val_dataset_indices.append(i)
            if test_annots:
                self.test_annotations.append(test_annots)
                self.test_dataset_names.append(self.dataset_names[i])
                self.test_dataset_indices.append(i)
        

        # filter for empty validation and test annotations (already filtered above, but keeping for safety)
        self.val_annotations = [annot for annot in self.val_annotations if annot]
        self.test_annotations = [annot for annot in self.test_annotations if annot]
        
        # Create dataloader_names mapping: maps dataloader_idx -> dataset_name (for val and test)
        # This will be used by callbacks to properly name metrics
        self.dataloader_names = {}
        for idx, name in enumerate(self.val_dataset_names):
            self.dataloader_names[idx] = name
        self.test_dataloader_names = {}
        for idx, name in enumerate(self.test_dataset_names):
            self.test_dataloader_names[idx] = name

        print(f"Number of training samples: {len(self.train_annotations)}")
        print(f"Number of validation samples: {sum([len(annot) for annot in self.val_annotations])} over {len(self.val_annotations)} datasets, {[len(annot) for annot in self.val_annotations]}")
        print(f"Number of test samples: {sum([len(annot) for annot in self.test_annotations])} over {len(self.test_annotations)} datasets, {[len(annot) for annot in self.test_annotations]}")
        print(f"Datasets: {self.dataset_names}")

    @property
    def names(self):
        """Mapping mode -> (dataloader_idx -> dataset name). Used by callbacks for log keys."""
        return {
            "val": getattr(self, "dataloader_names", None) or {},
            "test": getattr(self, "test_dataloader_names", None) or {},
        }

    def setup(self, stage: str) -> None:
        
        if stage != 'eval':
            self.train_dataset = TextAudioDataset(annotations=self.train_annotations, target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[0], root_dir=self.root_dirs[0], new_dir=self.new_dirs[0]) 
            
        self.val_datasets = [TextAudioDataset(annotations=self.val_annotations[i], target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[self.val_dataset_indices[i]], root_dir=self.root_dirs[self.val_dataset_indices[i]], new_dir=self.new_dirs[self.val_dataset_indices[i]]) for i in range(len(self.val_annotations))]
        self.test_datasets = [TextAudioDataset(annotations=self.test_annotations[i], target_n_samples=self.target_n_samples, target_sr=self.target_sr, return_audio=self.return_audio, return_text=self.return_text, concept=self.concept, preextracted_features=self.preextracted_features, truncate_preextracted=self.truncate_preextracted[self.test_dataset_indices[i]], root_dir=self.root_dirs[self.test_dataset_indices[i]], new_dir=self.new_dirs[self.test_dataset_indices[i]]) for i in range(len(self.test_annotations))]
        
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            drop_last=True  # Critical for distributed training to avoid hangs
        )
    
    def val_dataloader(self):
        return [
            DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False,
                drop_last=False  # Don't drop for validation
            ) 
            for val_dataset in self.val_datasets
        ]
    
    def test_dataloader(self):
        return [
            DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False,
                drop_last=False  # Don't drop for test
            ) 
            for test_dataset in self.test_datasets
        ]