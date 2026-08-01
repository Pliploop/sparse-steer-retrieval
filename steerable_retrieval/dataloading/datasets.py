from torch.utils.data import Dataset
from .loading_utils import *
import torch
import os
import random
from tqdm import tqdm
import pandas as pd
from hydra.utils import instantiate

import logging

class BasicProcessor:
    def __init__(self, probability = 1.0, split = None):
        self.probability = probability
        self.split = split
    
    def process(self, annot):
        """
        Process the annotation if the probability is less than the probability and the split is in the split list
        """
        if random.random() < self.probability:
            if self.split is None:
                return self.process(annot)
            if self.split is not None and annot['split'] in self.split:
                return self.process(annot)
            else:
                return annot
        else:
            return annot
        
    def __call__(self, annot):
        return self.process(annot)
    
class RandomNSentencesProcessor(BasicProcessor):

    def process(self, annot):
        
        prompt = annot['prompt']
        sentences = prompt.split('.')
        n_sentences = len(sentences)
        keep_n_sentences = random.randint(1, n_sentences)
        random_sentences = random.sample(random_sentences, keep_n_sentences)
        prompt = '. '.join(random_sentences)
        return prompt
    
class ShuffleSentencesProcessor(BasicProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def process(self, annot):
        prompt = annot['prompt']
        sentences = prompt.split('.')
        random.shuffle(sentences)
        prompt = '. '.join(sentences)
        return prompt

class ShuffleTagsProcessor(BasicProcessor):
    def __init__(self, replace_caption_p = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.replace_caption_p = replace_caption_p
    
    def process(self, annot):
        ## if tags is in the annotation, it will be a comma-separated string
        ## drop anywhere from all but 1 to 1 tag, and shuffle
        if 'tags' in annot:
            tags = annot['tags'].split(',')
        else:
            return annot
        len_tags = len(tags)
        drop_tags = random.randint(1, len_tags-1)
        tags_to_drop = random.sample(tags, drop_tags)
        tags = [tag for tag in tags if tag not in tags_to_drop]
        random.shuffle(tags)
        annot['tags'] = '. '.join(tags)

        # if replace caption, replace else add it to the caption
        if random.random() < self.replace_caption_p:
            annot['prompt'] = annot['tags']
        else:
            annot['prompt'] = annot['prompt'] + '.' + annot['tags']
        return annot

class TextAudioDataset(Dataset):
    def __init__(self,
                 annotations = None,
                 get_annotations_function = None,
                 task_kwargs = None,
                 target_n_samples = 96000,
                 target_sr = 48000,
                 return_audio = True,
                 return_text = True,
                 concept = None,
                 return_full_audio = False,
                 preextracted_features = False,
                 truncate_preextracted = 50,
                 split = None,
                 filter_split = None,
                 root_dir = None,
                 new_dir = None,
                 limit_n = None,
                 processors = [],
                 **kwargs
                 ):
        
        
        # Get annotations either directly or from function
        if annotations is not None:
            self.annotations = annotations
        elif get_annotations_function is not None:
            # Support both string and callable
            if isinstance(get_annotations_function, str):
                task_kwargs = task_kwargs or {}
                import importlib

                # parse the fully qualified function name
                module_name, func_name = get_annotations_function.rsplit('.', 1)
                module = importlib.import_module(module_name)
                get_annotations_func = getattr(module, func_name)
                self.annotations = get_annotations_func(**task_kwargs)
            else:
                task_kwargs = task_kwargs or {}
                self.annotations = get_annotations_function(**task_kwargs)
        else:
            raise ValueError("Must provide either annotations or get_annotations_function")
        



        self.target_n_samples = target_n_samples
        self.target_sr = target_sr
        self.return_audio = return_audio
        self.return_text = return_text
        self.concept = concept
        self.return_full_audio = return_full_audio
        self.preextracted_features = preextracted_features
        self.truncate_preextracted = truncate_preextracted
        self.split = split
        self.root_dir = root_dir
        self.new_dir = new_dir
        self.limit_n = limit_n
        # Update split if needed
        if split is not None and split != 'keep':
            for annot in self.annotations:
                annot['split'] = split
        elif split == 'keep':
            # Keep original splits from annotations, or set to 'train' if not present
            for annot in self.annotations:
                if 'split' not in annot or annot['split'] not in ['train', 'val', 'test']:
                    annot['split'] = 'train'
        elif split is None and len(self.annotations) > 0 and 'split' not in self.annotations[0].keys():
            for annot in self.annotations:
                annot['split'] = 'train'

        if filter_split is not None:
            self.annotations = [annot for annot in self.annotations if annot['split'] in filter_split]
                
        annot_df = pd.DataFrame(self.annotations)
        
        try:
            annot_df['file_index'] = pd.factorize(annot_df['file_path'])[0]
        except Exception as e:
            print(e)
        
        annot_df['file_path'] = annot_df['file_path'].apply(lambda x: x.replace(root_dir, new_dir) if root_dir is not None and new_dir is not None else x)
        
        self.annotations = annot_df.to_dict('records')

        # Filter out annotations whose feature/audio file is missing (avoids IndexError when retrying in __getitem__)
        if self.return_audio and self.preextracted_features:
            n_before = len(self.annotations)
            self.annotations = [
                a for a in self.annotations
                if os.path.exists(a['file_path'].replace('.mp3', '.npy').replace('.wav', '.npy'))
            ]
            if len(self.annotations) < n_before:
                logging.warning(
                    f"Filtered out {n_before - len(self.annotations)} annotations with missing feature files. "
                    f"Dataset has {len(self.annotations)} samples."
                )
        elif self.return_audio and not self.preextracted_features:
            n_before = len(self.annotations)
            def _audio_exists(a):
                p = a['file_path']
                return os.path.exists(p.replace('.npy', '.mp3')) or os.path.exists(p.replace('.npy', '.wav'))
            self.annotations = [a for a in self.annotations if _audio_exists(a)]
            if len(self.annotations) < n_before:
                logging.warning(
                    f"Filtered out {n_before - len(self.annotations)} annotations with missing audio files. "
                    f"Dataset has {len(self.annotations)} samples."
                )

        if self.limit_n is not None and self.limit_n < len(self.annotations):
            self.annotations = self.annotations[:self.limit_n]
            print(f"Limiting dataset to {self.limit_n} samples")
        else:
            print(f"Dataset has {len(self.annotations)} samples")

        
        assert return_audio or return_text, "At least one of return_audio or return_text must be True (duh)"

        self.processors = [instantiate(processor) for processor in processors]

        


    def purge(self):
        if self.return_audio and not self.preextracted_features:
            raise NotImplementedError("Purging your audio dataset is probably a bad idea")
        else:
            file_paths = [annot['file_path'] for annot in self.annotations]
            for file_path in file_paths:
                os.remove(file_path)
            print(f"Removed {len(file_paths)} files")
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, return_full_audio = False, hop = None, verbose = False):
        
        
        return_full_audio = self.return_full_audio if return_full_audio is None else return_full_audio
        
        annot = self.annotations[idx]
        
        
        if self.return_audio:
            if not self.preextracted_features:
                # file_path = annot['file_path'].replace('.npy','.mp3').replace('.wav','.mp3')
                # check if the mp3 file exists
                if os.path.exists(annot['file_path'].replace('.npy','.mp3')):
                    file_path = annot['file_path'].replace('.npy','.mp3')
                elif os.path.exists(annot['file_path'].replace('.npy','.wav')):
                    file_path = annot['file_path'].replace('.npy','.wav')
                else:
                    return self.__getitem__(idx+1)
                
                annot['file_path'] = file_path
                try:
                    audio = load_full_and_split(
                        annot['file_path'],
                        self.target_sr,
                        self.target_n_samples,
                        hop=hop,
                        verbose=verbose
                        ) if return_full_audio else load_audio_chunk(
                        annot['file_path'],
                        target_sr=self.target_sr,
                        target_n_samples=self.target_n_samples,
                        verbose=verbose
                        )
                    audio = audio.mean(1)
                except Exception as e:
                    return self.__getitem__(idx+1)
            else:
                file_path = annot['file_path'].replace('.mp3','.npy').replace('.wav','.npy')
                try:

                    audio = np.load(file_path,mmap_mode='r')
                    # Preextracted features may be stored per-clip as a single vector [D]
                    # or as multiple frames [T, D]; sample a random frame when framewise.
                    if audio.ndim > 1:
                        rand_index = random.randint(0, audio.shape[0]-1)
                        audio = audio[rand_index]
                    audio = torch.tensor(np.asarray(audio))
                except Exception as e:
                    return self.__getitem__(idx+1)
                
                
        
        if self.return_text:
            possible_captions = annot['caption']
            # ramdomly choose a caption hash
            
            random_hash = random.choice(list(possible_captions.keys()))
        
            caption = possible_captions[random_hash]
                    
        return_dict = {}
        
        if self.return_audio:
            # Clone so storage is resizable; avoids DataLoader collate error with mmap/numpy-derived tensors
            return_dict['audio'] = audio.clone() if isinstance(audio, torch.Tensor) else audio
            return_dict['file_path'] = annot['file_path']
            
        if self.return_text:
            return_dict['prompt'] = caption
                
        return_dict['file_idx'] = annot['file_index']

        for processor in self.processors:
            return_dict = processor(return_dict)
            
        return return_dict
    
    
    def extract_features(self, model, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding',hop = None, return_full_audio = True, verbose = False):
        
        device = next(model.parameters()).device
        print(f"Extracting features with {extract_method} method on {device} device") if verbose else None
        
        
        for param in model.parameters():
            param.requires_grad = False
        try:
            model.eval()
        except:
            pass
        
        for i in range(len(self)):
            try:
                item = self.__getitem__(i, return_full_audio = return_full_audio, hop = hop, verbose = verbose)
                file_path = self.annotations[i]['file_path'].replace('.mp3','.npy').replace('.wav','.npy')
                
                audio = item['audio'].squeeze(1).to(device)


                
                if audio.shape[0] > 200 :
                    chunks = torch.split(audio, 200, dim=0)
                    chunks = list(chunks)
                    audio_features = []
                    for chunk in chunks:
                        feat = getattr(model, extract_method)(chunk, **extract_kwargs)
                        if out_key is not None:
                            feat = feat[out_key]
                        audio_features.append(feat)
                    audio_features = torch.cat(audio_features, dim=0)
                else:
                    audio_features = getattr(model, extract_method)(audio.to(device), **extract_kwargs)
                    if out_key is not None:
                        audio_features = audio_features[out_key]
                
                    
                print(f"Extracted features for {file_path}, shape: {audio_features.shape}") if verbose else None
                
                yield audio_features, file_path
            except Exception as e:
                print(f"Error extracting features for {file_path}: {e}") if verbose else None
                continue
            
    def extract_and_save_features(self, model, save_dir = None, extract_method = 'extract_features', extract_kwargs = {}, out_key = 'embedding', hop = None, return_full_audio = True, limit_n = None, save = False, verbose = True, root_path = None, done_ids = None):
        
        
        print(self.__len__())
        
        audio_features_all = []
        counter = 0
        skipped_count = 0
        
        save_dir = '' if save_dir is None else save_dir
        done_ids = done_ids or set()
        
        if 's3://' in save_dir:
            import boto3
            import io
            client = boto3.client('s3')
        else:
            client = None
            import io

        # filter self.annotations to only include files that are not in done_ids
        new_annotations = []
        for annot in self.annotations:
            fp = annot['file_path']
            fp = fp.replace(root_path+'/','')
            # remove extension
            fp = fp.replace('.mp3','').replace('.wav','').replace('.npy','')

            if fp not in done_ids:
                new_annotations.append(annot)
        
        self.annotations = new_annotations
        
        for audio_features, file_path in (pbar:= tqdm(self.extract_features(model, extract_method = extract_method, extract_kwargs = extract_kwargs, out_key = out_key, hop = hop, return_full_audio = return_full_audio, verbose = verbose))):
            
            # print(file_path, root_path, save_dir)
            
            if root_path is not None:
                file_path = file_path.replace(root_path+'/','')

            save_path = os.path.join(save_dir, file_path)
            
            if save and audio_features is not None:
                

                #remove the root path from the file path
                
                
                if 's3://' in save_dir:
                    bucket, key = save_dir.replace("s3://", "").split("/", 1)
                    key = f"{key}/{file_path}"
                    
                    # local_path = os.path.join(local_temp_dir, file_path)
                    # os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    # np.save(local_path, audio_features.detach().cpu().numpy())
                    
                    pbar.set_description(f"Uploading features to s3://{bucket}/{key}") if verbose else None
                    try:
                        # client.upload_file(save_path, bucket, key)
                        
                        buffer = io.BytesIO()
                        np.save(buffer, audio_features.detach().cpu().numpy())
                        buffer.seek(0)
                        client.put_object(Bucket=bucket, Key=key, Body=buffer)
                    except Exception as e:
                        print(f"Error uploading to s3: {e}") if verbose else None
                        
                    # os.remove(local_path)
                else:
                    pbar.set_description(f"Saving features in {save_path}, shape: {audio_features.shape}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    np.save(save_path, audio_features.detach().cpu().numpy())
            
            if not save and audio_features is not None:
                pbar.set_description(f"{file_path}, shape: {audio_features.shape}")
                pass
                
            audio_features_all.append(audio_features.detach().cpu()) if audio_features is not None else None
            
            counter += 1
            if limit_n and counter >= limit_n:
                break
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed items") if verbose else None
        
        try:   
            print(f"Returning {len(audio_features_all)} features") if verbose else None
            all_= torch.stack(audio_features_all)
            print(f"Stacked features, shape: {all_.shape}") if verbose else None
            return all_
        
        
        except:
            return None