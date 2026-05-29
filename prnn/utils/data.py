import numpy as np
from numpy.random import randint
import copy
import torch
import shutil

from shutil import copytree, ignore_patterns
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class TrajDataset(Dataset):
    def __init__(self, folder: str, seq_length: int, n_trajs: int,
                 act_datatype=None, n_obs=1):
        self._data_dir = folder
        #self.path = Path(folder)
        self.n_trajs = n_trajs
        self.seq_length = seq_length
        self.act_type = act_datatype # different depending on the environment
        self.n_obs = n_obs
        self.raw = False

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        act = np.load(self._data_dir + '/' + str(index+1) + "/act.npy")[:,:self.seq_length]
        act = torch.tensor(act, dtype=self.act_type)
        if self.n_obs > 1: # for multimodal observations
            obs = [np.load(self._data_dir + '/' + str(index+1) + "/obs%d.npy" % i)[:,:self.seq_length+1] for i in range(self.n_obs)]
            obs = [torch.tensor(obs[i], dtype=torch.float32) for i in range(self.n_obs)]
            return *obs, act
        else:
            obs = np.load(self._data_dir + '/' + str(index+1) + "/obs.npy")[:,:self.seq_length+1]
            obs = torch.tensor(obs, dtype=torch.float32)
            return obs, act

class TrajRawDataset(TrajDataset):
    def __init__(self, folder: str, seq_length: int, n_trajs: int,
                 act_datatype=None, n_obs=1, act=True):
        super().__init__(folder, seq_length, n_trajs, act_datatype, n_obs)
        self.raw = True
        self.act = act

    def __getitem__(self, index):
        raw = np.load(self._data_dir + '/' + str(index+1) + "/raw.npy")[:self.seq_length+1]
        raw = torch.tensor(raw, dtype=torch.float32)
        if not self.act:
            return raw
        act = np.load(self._data_dir + '/' + str(index+1) + "/act.npy")[:,:self.seq_length]
        act = torch.tensor(act, dtype=self.act_type)
        return raw, act


class RandomRawDataset(TrajDataset):
    def __init__(self, folders: str or list, seq_length: int, n_trajs: int,
                 act_datatype=None, n_obs=0):
        super().__init__(folders, seq_length, n_trajs, act_datatype, n_obs)
        self.raw = True
        self._raw_files = []

        if isinstance(folders, str):
            folders = [folders]
        traj_dirs = []
        for folder in folders:
            for child in Path(folder).iterdir():
                if child.is_dir():
                    traj_dirs.append(child)
        traj_dirs.sort(key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name))

        raw_length = None
        for traj_dir in traj_dirs:
            raw_path = traj_dir / "raw.npy"
            if raw_path.exists():
                raw = np.load(str(raw_path), mmap_mode='r')
                if raw_length is None:
                    raw_length = raw.shape[0]
                elif raw.shape[0] != raw_length:
                    raise ValueError(
                        f"All raw.npy files must have the same length, but {raw_path} has length {raw.shape[0]} "
                        f"while the expected length is {raw_length}."
                    )
                self._raw_files.append(raw_path)

        if not self._raw_files:
            raise FileNotFoundError(f"No raw.npy files found in {folders}")

        self._raw_length = raw_length

    def __len__(self):
        return len(self._raw_files) * self._raw_length

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        traj_idx = index // self._raw_length
        raw_idx = index % self._raw_length

        raw = np.load(str(self._raw_files[traj_idx]), mmap_mode='r')[raw_idx]
        raw = torch.tensor(raw, dtype=torch.float32)
        return raw

def generate_trajectories(
            env,
            agent,
            n_trajs,
            seq_length,
            folder,
            save_raw = False,
            continuous_trajectories = False
        ):
    
    top_dir = Path(folder)
    n_generated = 0
    length_generated = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(env, 'encoder'):
        env.encoder.to(device)

    # If there is a folder with the same name as the environment,
    # check how many trajectories and of what length are already generated
    if top_dir.exists():
        print("Found existing data, will generate more data if needed")
        for child in top_dir.iterdir():
            n_generated += 1
        try:
            length_generated = len(np.load(str(top_dir / "1/act.npy"))[0])
        except:
            pass
    else:
        top_dir.mkdir(parents=True)

    # If the generated trajectories are shorter than the desired length,
    # start from the last state and generate the rest of the trajectory
    if length_generated and length_generated < seq_length:
        print('Sequence length is shorter than desired, generating more data...')
        for child in top_dir.iterdir():
            if child.is_dir():
                state = np.load(str(child / "state.npy"))
                env.load_state(state)
                act = np.load(str(child / "act.npy"))
                new_data = env.collectObservationSequence(
                        agent,
                        seq_length - length_generated,
                        obs_format = 'npgrid',
                        reset = not continuous_trajectories,
                        save_env = save_raw,
                        device = device
                    )
                new_obs, new_act, new_state = new_data[0], new_data[1], new_data[2]
                if save_raw:
                    raw = np.load(str(child / "raw.npy"))
                    new_raw = new_data[4]
                    raw = np.concatenate((raw, new_raw), axis=1)
                    np.save(str(child / "raw.npy"), raw)
                if env.n_obs > 1: # for multimodal observations
                    for i in range(env.n_obs):
                        obs = np.load(str(child / ("obs"+str(i)+".npy")))
                        obs = np.concatenate((obs, new_obs[i][:,1:]), axis=1)
                        np.save(str(child / ("obs"+str(i)+".npy")), obs)
                else:
                    obs = np.load(str(child / "obs.npy"))
                    obs = np.concatenate((obs, new_obs[:,1:]), axis=1)
                    np.save(str(child / "obs.npy"), obs)

                act = np.concatenate((act, new_act), axis=1)
                np.save(str(child / "act.npy"), act)
                last_state = env.save_state(new_state)
                np.save(str(child / "state.npy"), last_state)
    
    # Generate new trajectories in addition to the already existing ones
    if n_generated < n_trajs:
        print('Not enough trajectories, generating more data...')
        for i in range(n_trajs - n_generated):
            traj_dir = top_dir / str(n_generated + i + 1)
            traj_dir.mkdir()
            data = env.collectObservationSequence(agent, seq_length, obs_format='npgrid',
                                                  save_env=save_raw, device=device)
            obs, act, state = data[0], data[1], data[2]
            if save_raw:
                raw_data = data[4]
                np.save(str(traj_dir / "raw.npy"), raw_data)
            last_state = env.save_state(state)
            if env.n_obs > 1: # for multimodal observations
                for j in range(env.n_obs):
                    np.save(str(traj_dir / ("obs"+str(j)+".npy")), obs[j])
            else:
                np.save(str(traj_dir / "obs.npy"), obs)
            np.save(str(traj_dir / "act.npy"), act)
            np.save(str(traj_dir / "state.npy"), last_state)

    
    if hasattr(env, 'encoder'):
        env.encoder.to('cpu')


def create_dataloader(env, agent, n_trajs, seq_length, folder,
                      generate=True, tmp_folder=None, batch_size=32,
                      num_workers=0, save_raw=False, load_raw=False,
                      random_raw=False, load_act=True):
    folder = folder + '/' + env.name + '-' + agent.name + '-' + env.act_enc 
    if generate:
        generate_trajectories(env, agent, n_trajs, seq_length, folder, save_raw=save_raw)
    if not tmp_folder:
        tmp_folder = folder
    else:
        tmp_folder = tmp_folder + '/' + env.name + '-' + agent.name + '-' + env.act_enc 
        if random_raw:
            copytree(folder, tmp_folder, dirs_exist_ok=True, ignore=ignore_patterns("obs*", "act.npy", "state.npy"))
        elif not load_raw:
            copytree(folder, tmp_folder, dirs_exist_ok=True, ignore=ignore_patterns("raw.npy", "state.npy"))
        else:
            copytree(folder, tmp_folder, dirs_exist_ok=True, ignore=ignore_patterns("obs*", "state.npy"))
    if random_raw: # raw images for training the encoder, drawn randomly
        dataset = RandomRawDataset(tmp_folder, seq_length, n_trajs, env.getActType(), env.n_obs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif not load_raw:
        dataset = TrajDataset(tmp_folder, seq_length, n_trajs, env.getActType(), env.n_obs)
    elif not load_act: # raw images for training the encoder, drawn in sequences
        dataset = TrajRawDataset(tmp_folder, seq_length, n_trajs, act=False)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        dataset = TrajRawDataset(tmp_folder, seq_length, n_trajs, env.getActType(), env.n_obs)
    env.addDataLoader(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))
    print(f"Dataloader created with {n_trajs} trajectories, sequence length {seq_length}")

    

    
    
class MergedTrajDataset(TrajDataset):
    def __init__(self, folders: list, seq_length: int, n_trajs: list,
                 act_datatype=None, n_obs=1, raw=False):
        super().__init__(folders, seq_length, n_trajs, act_datatype)
        self.raw = raw
    def __len__(self):
        return sum(self.n_trajs)
    def __getitem__(self, index):
        #Try finding the traj in the folder, if the index is higher than n_trajs for that folder, go to the next one
        for i, n in enumerate(self.n_trajs):
            try:
                act = np.load(self._data_dir[i] + '/' + str(index+1) + "/act.npy")[:,:self.seq_length]
                if not self.raw:
                    obs = np.load(self._data_dir[i] + '/' + str(index+1) + "/obs.npy")[:,:self.seq_length+1]
                else:
                    obs = np.load(self._data_dir[i] + '/' + str(index+1) + "/raw.npy")[:self.seq_length+1]
                act = torch.tensor(act, dtype=self.act_type)
                obs = torch.tensor(obs, dtype=torch.float32)
                break
            except FileNotFoundError:
                index-=n
        return obs, act

def mergeDatasets(envs, batch_size=1, shuffle=True, num_workers=0, mixed_batch=True, raw=False):
    datafolders = [env.dataLoader.dataset._data_dir for env in envs]
    seq_length = [env.dataLoader.dataset.seq_length for env in envs]
    n_trajs = [env.dataLoader.dataset.n_trajs for env in envs]
    env_act_datatype = envs[0].getActType()
    for env in envs:
        assert env.getActType() == env_act_datatype
    datasetMerged = MergedTrajDataset(datafolders, min(seq_length), n_trajs, env_act_datatype, raw=raw)

    iterators = [env.killIterator() for env in envs]
    envMerged = copy.deepcopy(envs[0])
    for i,env in enumerate(envs):
        env.DL_iterator = iterators[i]
    
    if not mixed_batch:
        indices = num_to_indices(n_trajs)
        batch_sampler = GroupedBatchRandomSampler(indices, batch_size)
        envMerged.addDataLoader(DataLoader(datasetMerged, batch_sampler=batch_sampler,
                                        num_workers=num_workers))
    else:
        envMerged.addDataLoader(DataLoader(datasetMerged, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers))
    return envMerged


def num_to_indices(n_trajs):
    indices_list = []
    start_index = 0
    for length in n_trajs:
        indices = list(range(start_index, start_index + length))
        indices_list.append(indices)
        start_index += length
    return indices_list

from torch.utils.data.sampler import Sampler
import random
class GroupedBatchRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, indices, batch_size, generator=None) -> None:
        #indices: list of lists of indices (groups)
        self.indices = indices
        self.generator = generator
        self.batch_size = batch_size

    def __iter__(self):
        #Random permute each group of indices
        indices = [random.sample(inds,len(inds)) for inds in self.indices]
        #Break each list into groups of size batch_size
        indices = [[inds[i:i + self.batch_size] for i in range(0, len(inds), self.batch_size)] for inds in indices]
        #Merge and Randomly permute the lists
        indices = sum(indices, [])
        random.shuffle(indices)

        for i in torch.randperm(len(indices), generator=self.generator):
            #If it's smaller than batch size, drop it and continue
            if len(indices[i])==self.batch_size:
                yield indices[i]

    def __len__(self) -> int:
        numbatches = [len(i) // self.batch_size for i in self.indices]
        return sum(numbatches)
