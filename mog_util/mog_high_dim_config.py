import torch

# identical
# d = 10  # Dimensionality
# pis = torch.tensor([0.5, 0.5]).to('cuda') # the probability of each gaussian. pis.sum() should be 1
# mus = [torch.tensor((0.3,) * d).to('cuda'), torch.tensor((-0.3,) * d).to('cuda')]
# sigmas = [0.3 * torch.eye(d).to('cuda'), 0.3 * torch.eye(d).to('cuda')]

# 一圈高斯
# 高斯数量和维度
def mog_high_dim_config():
    k = 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.05 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    # mean_off_set = torch.ones((k, d)).to('cuda') * 0.3 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    # pis = torch.tensor([1,1,1,10,10,10]).to('cuda')
    pis = pis / pis.sum()
    grad_log_p_noise = 0.25
    log_p_noise = 0

    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    angles = torch.linspace(0, 2 * torch.pi, k + 1)[:k]
    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise


def mog_high_dim_config2():
    mode_per_mode = 3
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    # sigma = 0.007 # 每个高斯的标准差
    sigma = 0.007 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = pis / pis.sum()
    grad_log_p_noise = 0
    log_p_noise = 0


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 3
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise


def mog_high_dim_config2_noise():
    mode_per_mode = 3
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.007 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = pis / pis.sum()
    grad_log_p_noise = 0.05
    log_p_noise = 0.05


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 3
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise

def mog_high_dim_config3():
    mode_per_mode = 3
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.007 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = pis / pis.sum()
    grad_log_p_noise = 0
    log_p_noise = 0


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 5
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        # diagonal[:2] = diagonal[:2] * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise

def mog_high_dim_config4():
    mode_per_mode = 3
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.007 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = pis / pis.sum()
    grad_log_p_noise = 0
    log_p_noise = 0


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 5
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        if (i + 2) % 3 != 0:
            diagonal = torch.ones(d).to('cuda') * sigma * 0.5
        else:
            diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise

def mog_high_dim_config5():
    mode_per_mode = 2
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.007 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    # pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = torch.tensor((1, 5, ) * 6).to('cuda')
    pis = pis / pis.sum()
    grad_log_p_noise = 0
    log_p_noise = 0


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 7
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise

def mog_high_dim_config6_noise():
    mode_per_mode = 3
    k = mode_per_mode * 6 # 高斯的数量
    d = 10 # 合成数据的维度
    radius = 1  # 圆的半径
    sigma = 0.01 # 每个高斯的标准差
    mean_off_set = torch.ones((k, d)).to('cuda') * 0 # 对分布进行偏移
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    pis = pis / pis.sum()
    grad_log_p_noise = 0.01
    log_p_noise = 0.001


    # 初始化存储参数的张量
    mus = torch.zeros((k, d)).to('cuda')
    sigmas = torch.zeros((k, d, d)).to('cuda')
    # 生成2D平面上均匀分布的角度
    _angles = torch.linspace(0, 2 * torch.pi, int(k / mode_per_mode) + 1)[:int(k / mode_per_mode)]
    angles = []
    _angle = 2 * torch.pi / int(k / mode_per_mode) / 3
    for angle in _angles:
        angles.append(angle - _angle)
        angles.append(angle)
        angles.append(angle + _angle)

    # 生成 means
    for i in range(k):
        x = radius * torch.cos(angles[i])
        y = radius * torch.sin(angles[i])
        # 在2D平面上均匀分布，并在剩余维度上添加小的随机偏移
        mus[i, :2] = torch.tensor([x, y]).to('cuda')
        # mus[i, 2:] = torch.randn(8) * 0.1  # 剩余维度的小随机偏移
        mus[i, 2:] = 0
    mus = mus + mean_off_set

    # 生成 covariance matrices
    for i in range(k):
        # 对角线上的元素表示每个维度上的方差
        # diagonal = torch.rand(d) * 0.5 + 0.5  # 范围在[0.5, 1.0]之间
        diagonal = torch.ones(d).to('cuda') * sigma
        sigmas[i] = torch.diag(diagonal)
    return mus, sigmas,d,  pis, grad_log_p_noise, log_p_noise

def mog_chessboard_high_dim():
    d = 10
    grid_size = 4
    grad_log_p_noise, log_p_noise = 0, 0

    index = 0
    mus = []
    cell_length = 1 / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:  # Black cell condition
                # Calculate the origin of the cell
                mean = torch.tensor((0.,) * d).to('cuda')
                origin_x = i * cell_length
                origin_y = j * cell_length
                mean[:2] = torch.tensor([origin_x, origin_y]).to('cuda')
                mus.append(mean)
                index += 1
    mus = torch.stack(mus, dim=0)
    mus = mus - torch.mean(mus, dim=0, keepdim=True)

    k = len(mus)
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    sigmas = torch.zeros((k, d, d)).to('cuda')

    sigma_scale = 0.007
    for i in range(k):
        diagonal = torch.ones(d).to('cuda') * sigma_scale
        sigmas[i] = torch.diag(diagonal)

    return mus, sigmas, d,  pis, grad_log_p_noise, log_p_noise

def spiral():
    d = 10
    k = 16
    mus = torch.zeros((k, d)).to('cuda')
    grad_log_p_noise, log_p_noise = 0, 0
    num_turns = 2
    max_radius = 1

    # 生成等间距的角度值，覆盖多个360度圈
    angles = torch.linspace(0, num_turns * 2 * torch.pi, steps=k).to('cuda')

    # 半径随角度增大而线性增大
    radii = torch.linspace(0, max_radius, steps=k).to('cuda')

    # 将极坐标转换为笛卡尔坐标
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    mus[:, 0] = x
    mus[:, 1] = y

    k = len(mus)
    pis = torch.ones(k).to('cuda') / k  # 每个高斯的权重
    sigmas = torch.zeros((k, d, d)).to('cuda')

    sigma_scale = 0.007
    for i in range(k):
        diagonal = torch.ones(d).to('cuda') * sigma_scale
        sigmas[i] = torch.diag(diagonal)

    return mus, sigmas, d,  pis, grad_log_p_noise, log_p_noise

# mus, sigmas,d, pis, grad_log_p_noise, log_p_noise = mog_high_dim_config2_noise()
# mus, sigmas,d, pis, grad_log_p_noise, log_p_noise = mog_high_dim_config2()
mus, sigmas,d, pis, grad_log_p_noise, log_p_noise = mog_chessboard_high_dim()
# mus, sigmas,d, pis, grad_log_p_noise, log_p_noise = spiral()
ground_truth_num = 5000
synthetic_num = 5000