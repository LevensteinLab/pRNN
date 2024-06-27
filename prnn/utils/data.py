import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class TrajDataset(Dataset):
    def __init__(self, folder: str, seq_length: int, n_trajs: int):
        self._data_dir = folder
        self.path = Path(folder)
        self.n_trajs = n_trajs
        self.seq_length = seq_length
        # self.transform = ToTensor()

        # acts = []
        # obss = []
        # for i in range(self.n_trajs):
        #     acts.append(np.load(self._data_dir + '/' + str(i+1) + "/act.npy")[:seq_length])
        #     obss.append(np.load(self._data_dir + '/' + str(i+1) + "/obs.npy")[:seq_length+1])
        # self.acts = np.concatenate(acts, axis=0)
        # self.obss = np.concatenate(obss, axis=0)

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        # act = self.acts[index]
        # obs = self.obss[index]
        act = np.load(self._data_dir + '/' + str(index+1) + "/act.npy")[0,:self.seq_length]
        obs = np.load(self._data_dir + '/' + str(index+1) + "/obs.npy")[0,:self.seq_length+1]
        act = torch.tensor(act, dtype=torch.int64)
        obs = torch.tensor(obs, dtype=torch.float32)
        return obs, act


def generate_trajectories(env, agent, n_trajs, seq_length, folder):
    top_dir = Path(folder)
    n_generated = 0
    length_generated = 0

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
                obs = np.load(str(child / "obs.npy"))
                act = np.load(str(child / "act.npy"))
                new_obs, new_act, new_state, _ = env.collectObservationSequence(agent,
                                                                                seq_length - length_generated,
                                                                                obs_format='npgrid',
                                                                                reset=False)
                obs = np.concatenate((obs, new_obs[:,1:]), axis=1)
                act = np.concatenate((act, new_act), axis=1)
                last_state = env.save_state(act, new_state)
                np.save(str(child / "obs.npy"), obs)
                np.save(str(child / "act.npy"), act)
                np.save(str(child / "state.npy"), last_state)
    
    # Generate new trajectories in addition to the already existing ones
    if n_generated < n_trajs:
        print('Not enough trajectories, generating more data...')
        for i in range(n_trajs - n_generated):
            traj_dir = top_dir / str(n_generated + i + 1)
            traj_dir.mkdir()
            obs, act, state, _ = env.collectObservationSequence(agent, seq_length, obs_format='npgrid')
            last_state = env.save_state(act, state)
            np.save(str(traj_dir / "obs.npy"), obs)
            np.save(str(traj_dir / "act.npy"), act)
            np.save(str(traj_dir / "state.npy"), last_state)


def create_dataloader(env, agent, n_trajs, seq_length, folder, batch_size=32, num_workers=0):
    folder = folder + '/' + env.name + '-' + type(agent).__name__
    generate_trajectories(env, agent, n_trajs, seq_length, folder)
    dataset = TrajDataset(folder, seq_length, n_trajs)
    env.addDataLoader(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))

    
