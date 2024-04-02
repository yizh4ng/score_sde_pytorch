import numpy as np
# experiment = 'ddpmpp_vp_deep_continuous_euler_maruyama'
# experiment = 'ddpmpp_vp_deep_continuous_adaptive'
experiment = 'ddpmpp_vp_continuous_ddim'
# experiment = 'ddpm_vp_adaptive'
path = f'/home/yi/workplace/score_sde_pytorch/{experiment}/eval/report_8.npz'

data = np.load(path)
lst = data.files
for item in lst:
    print(item)
    print(data[item])