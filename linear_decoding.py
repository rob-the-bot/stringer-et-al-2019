# %%
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import decoders

### WHERE YOU DOWNLOADED THE FIGSHARE
dataroot = Path('Z:/') / "Tianfeng_workspace" / "PCA_LDA" / "data" / "stringer2019" / "fs125"

# file list
# THIS WAS UPDATED ON NOV 5th!!! Please redownload + biased_V2 and static_sin_rand
db = np.load(dataroot / 'database.npy', allow_pickle=True)

fs = []
for di in db:
    fname = f"{di['expt']}_{di['mouse_name']}_{di['date']}_{di['block']}.npy"
    fs.append(dataroot / fname)

### WHERE YOU WANT TO SAVE THE OUTPUTS OF THE ANALYSIS SCRIPTS AND THE FIGURES (if save_figure=True)
saveroot = dataroot / 'figs'

#%% linear decoding from all neurons

for percent in tqdm(np.logspace(-2, 0, 10)):
    if percent == 1:
        percent = None
        fname = 'linear_decoder_asymp_all.npy'
    else:
        fname = f'linear_decoder_asymp_percent_{percent:.2f}.npy'
    E, ccE, nsplit, npop, nstim, E2 = decoders.asymptotics(fs[:6], linear=True, downsample=percent, skip_E2=True)
    np.save(saveroot / fname, {'E': E})
