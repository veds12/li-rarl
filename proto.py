import pickle

with open('/network/scratch/v/vedant.shah/li-rarl/data/expert_trajs/babyai_demos-36-GoToLocal-100000_seq2seq_dt.pkl', 'rb') as f:
    traj = pickle.load(f)
    f.close()

print(traj['images'][0].shape)