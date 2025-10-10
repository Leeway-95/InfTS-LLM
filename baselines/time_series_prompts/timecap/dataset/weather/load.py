import pickle as pkl

with open('indices.pkl', 'rb') as f:
    indices = pkl.load(f)

with open('dates.pkl', 'rb') as f:
    dates = pkl.load(f)

for city in ['hs', 'ny', 'sf']:
    with open(f'time_series_{city}.pkl', 'rb') as f:
        data = pkl.load(f)

    with open(f'rain_{city}.pkl', 'rb') as f:
        rain = pkl.load(f)