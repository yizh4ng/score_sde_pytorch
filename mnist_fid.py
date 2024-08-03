import torch
import torchvision
import torchvision.models as models
import numpy as np
from PIL import Image
from scipy import linalg

from mnist_fid_util import evaluate_fid_score

"""
FID 测试一般3000~5000张图片，
FID小于50：生成质量较好，可以认为生成的图像与真实图像相似度较高。
FID在50到100之间：生成质量一般，生成的图像与真实图像相似度一般。
FID大于100：生成质量较差，生成的图像与真实图像相似度较低。
"""


# 加载预训练inception v3模型, 并移除top层，第一次运行会下载模型到cache里面
def load_inception():
    model = models.inception_v3(weights='IMAGENET1K_V1').to('cuda')
    model.eval()
    # 将fc用Identity()代替，即去掉fc层
    model.fc = torch.nn.Identity()
    return model


# inception v3 特征提取
def extract_features(images, model):
    # images = images / 255.0
    with torch.no_grad():
        feat = model(images)
    return feat.cpu().numpy()


# FID计算
def cal_fid(images1, images2):
    """
    images1, images2: nchw 归一化，且维度resize到[N,3,299,299]
    """
    model = load_inception()

    # 1. inception v3 特征
    feats1 = extract_features(images1, model)
    feats2 = extract_features(images2, model)

    # 2. 均值协方差
    feat1_mean, feat1_cov = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    feat2_mean, feat2_cov = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    # 3. Fréchet距离
    sqrt_trace_cov = linalg.sqrtm(feat1_cov @ feat2_cov)
    fid = np.sum((feat1_mean - feat2_mean) ** 2) + np.trace(feat1_cov + feat2_cov - 2 * sqrt_trace_cov)
    return fid.real


def cal_feature_fid(feats1, feats2):
    # 2. 均值协方差
    feat1_mean, feat1_cov = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    feat2_mean, feat2_cov = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)

    # 3. Fréchet距离
    sqrt_trace_cov = linalg.sqrtm(feat1_cov @ feat2_cov)
    fid = np.sum((feat1_mean - feat2_mean) ** 2) + np.trace(feat1_cov + feat2_cov - 2 * sqrt_trace_cov)
    return fid.real


if __name__ == '__main__':
    # f = cal_fid(torch.rand(1000, 3, 299, 299).to('cuda'), torch.rand(1000, 3, 299, 299).to('cuda'))
    # print(f)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    mnist_dset = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)
    mnist_loader = torch.utils.data.DataLoader(mnist_dset, batch_size=1000, shuffle=True)

    # 获取一个batch的MNIST数据
    mnist_images, _ = next(iter(mnist_loader))
    mnist_images = (mnist_images + 1) * 0.5
    # mnist_images = mnist_images.to('cuda')

    # # 提取特征
    # model = load_inception()
    # mnist_features = extract_features(mnist_images, model)
    # print(mnist_features.shape)

    # 从 .npz 文件加载生成的样本
    data = np.load("./ula.npz")
    generated_samples = data['samples']

    # 转换生成的样本并提取特征
    generated_samples = np.stack(
        [np.array(transforms(Image.fromarray((sample * 255).astype(np.uint8)))) for sample in
         generated_samples])
    generated_samples = (generated_samples + 1) * 0.5

    # generated_samples = torch.tensor(generated_samples).to('cuda')
    #
    # generated_features = extract_features(generated_samples, model)
    # print("Generated features shape:", generated_features.shape)
    # fid = cal_feature_fid(mnist_features, generated_features)
    # print(fid)
    fid = evaluate_fid_score(mnist_images, generated_samples, batch_size=1000)
    print(fid)