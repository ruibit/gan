#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_wgan.py
# Author            : none <none>
# Date              : 14.04.2022
# Last Modified Date: 15.04.2022
# Last Modified By  : none <none>
""" 基于MNIST 实现Wasserstein GAN (WGAN) """

import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

# 定义图像尺寸和潜在空间维度
image_size = [1, 28, 28]  # MNIST图像尺寸: 1通道, 28x28像素
latent_dim = 100  # 生成器输入噪声向量的维度
label_emb_dim = 10
batch_size = 64  # 训练批量大小
use_gpu = torch.cuda.is_available()  # 检查是否可用GPU
save_dir = "wgan_generated_images"
os.makedirs(save_dir, exist_ok=True)

# 创建目录用于保存损失图像
loss_plot_dir = "wgan_loss_plots"
os.makedirs(loss_plot_dir, exist_ok=True)

# WGAN关键参数
n_critic = 5  # 判别器训练次数:生成器训练次数的比例
clip_value = 0.01  # 权重裁剪值


class Generator(nn.Module):
    """生成器网络 - 使用转置卷积将随机噪声转换为逼真的图像"""

    def __init__(self):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(10, label_emb_dim)

        # 初始全连接层，将噪声向量映射到卷积特征图
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(True)
        )

        # 转置卷积层，逐步上采样到目标图像尺寸
        self.deconv_layers = nn.Sequential(
            # 输入: (batch_size, 128, 7, 7)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 输出: (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # 输出: (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False),  # 输出: (1, 28, 28)
            nn.Tanh()  # 输出范围[-1,1]，与标准化后的图像数据匹配
        )

    def forward(self, z, labels):
        # 输入z: [batch_size, latent_dim] 随机噪声向量
        # 输入labels: [batch_size] 标签

        # 嵌入标签并与噪声拼接
        label_embedding = self.embedding(labels)  # [batch_size, label_emb_dim]
        z = torch.cat([z, label_embedding], dim=1)  # [batch_size, latent_dim + label_emb_dim]

        # 通过全连接层并重塑为特征图
        x = self.fc(z)  # [batch_size, 7*7*128]
        x = x.view(-1, 128, 7, 7)  # [batch_size, 128, 7, 7]

        # 通过转置卷积层生成图像
        image = self.deconv_layers(x)  # [batch_size, 1, 28, 28]

        return image


class Discriminator(nn.Module):
    """判别器网络 - WGAN使用Critic而不是判别器，输出分数而不是概率"""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(10, label_emb_dim)

        # 卷积层，逐步下采样提取特征
        self.conv_layers = nn.Sequential(
            # 输入: (batch_size, 1, 28, 28)
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),  # 输出: (32, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 输出: (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # 输出: (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 全连接层输出分数（WGAN不需要Sigmoid）
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7 + label_emb_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
            # 注意：WGAN移除了Sigmoid，输出实数分数
        )

    def forward(self, image, labels):
        # 输入图像: [batch_size, 1, 28, 28]
        # 输入labels: [batch_size] 标签

        # 通过卷积层提取特征
        features = self.conv_layers(image)  # [batch_size, 128, 7, 7]
        features = features.view(features.size(0), -1)  # [batch_size, 128*7*7]

        # 嵌入标签并与特征拼接
        label_embedding = self.embedding(labels)  # [batch_size, label_emb_dim]
        combined = torch.cat([features, label_embedding], dim=1)  # [batch_size, 128*7*7 + label_emb_dim]

        # 通过全连接层输出分数
        score = self.fc(combined)  # [batch_size, 1]

        return score


# 数据准备
# 加载MNIST数据集，使用标准化将图像范围调整到[-1,1]以匹配Tanh输出
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1,1]
                                         ]
                                     )
                                     )
# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 定义优化器 - WGAN使用RMSprop而不是Adam
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0003)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0003)

# 如果可用，将模型和数据移动到GPU
if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()

# 初始化损失记录列表
g_losses = []
d_losses = []
wasserstein_distances = []  # WGAN特有的Wasserstein距离度量
steps = []


# 权重裁剪函数
def clip_weights(model, clip_value):
    for param in model.parameters():
        param.data.clamp_(-clip_value, clip_value)


# 训练循环
num_epoch = 200  # 训练轮数
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, labels = mini_batch  # 获取真实图像和标签

        # 将数据移动到GPU（如果可用）
        if use_gpu:
            gt_images = gt_images.to("cuda")
            labels = labels.to("cuda")

        # 训练判别器 (Critic) n_critic 次
        for _ in range(n_critic):
            # 生成随机噪声作为生成器输入
            z = torch.randn(batch_size, latent_dim)
            if use_gpu:
                z = z.to("cuda")

            # 清零判别器梯度
            d_optimizer.zero_grad()

            # 计算真实图像的分数
            real_scores = discriminator(gt_images, labels)

            # 生成假图像
            fake_images = generator(z, labels)

            # 计算假图像的分数
            fake_scores = discriminator(fake_images.detach(), labels)

            # WGAN判别器损失：最大化真实图像分数与假图像分数的差距
            d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)

            # 反向传播和优化
            d_loss.backward()
            d_optimizer.step()

            # 裁剪判别器权重以满足Lipschitz约束
            clip_weights(discriminator, clip_value)

        # 训练生成器
        # 生成随机噪声
        z = torch.randn(batch_size, latent_dim)
        if use_gpu:
            z = z.to("cuda")

        # 清零生成器梯度
        g_optimizer.zero_grad()

        # 生成假图像
        fake_images = generator(z, labels)

        # 计算假图像的分数
        fake_scores = discriminator(fake_images, labels)

        # WGAN生成器损失：最小化假图像分数的负值（相当于最大化假图像分数）
        g_loss = -torch.mean(fake_scores)

        # 反向传播和优化
        g_loss.backward()
        g_optimizer.step()

        # 计算Wasserstein距离（用于监控训练进度）
        wasserstein_distance = torch.mean(real_scores) - torch.mean(fake_scores)

        # 记录损失值
        current_step = len(dataloader) * epoch + i
        steps.append(current_step)
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        wasserstein_distances.append(wasserstein_distance.item())

        # 定期输出训练状态
        if i % 50 == 0:
            print(
                f"step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item():.4f}, d_loss:{d_loss.item():.4f}, wasserstein_dist:{wasserstein_distance.item():.4f}")

        # 定期保存生成的图像样本
        if i % 400 == 0:
            # 将图像从[-1,1]范围转换回[0,1]范围以便保存
            image = (fake_images[:16].data + 1) / 2
            torchvision.utils.save_image(image, os.path.join(save_dir, f"image_{len(dataloader) * epoch + i}.png"),
                                         nrow=4)

# 训练结束后绘制最终损失图像
print("Training completed! Generating final loss plots...")
plt.figure(figsize=(15, 10))

# 绘制生成器和判别器损失
plt.subplot(2, 2, 1)
plt.plot(steps, g_losses, label='Generator Loss', color='blue')
plt.plot(steps, d_losses, label='Discriminator Loss', color='red')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('WGAN Generator and Discriminator Loss')
plt.legend()
plt.grid(True)

# 绘制Wasserstein距离
plt.subplot(2, 2, 2)
plt.plot(steps, wasserstein_distances, label='Wasserstein Distance', color='green')
plt.xlabel('Training Steps')
plt.ylabel('Distance')
plt.title('Wasserstein Distance')
plt.legend()
plt.grid(True)

# 绘制损失比率
plt.subplot(2, 2, 3)
ratio = [g / d if d != 0 else 0 for g, d in zip(g_losses, d_losses)]
plt.plot(steps, ratio, label='G/D Loss Ratio', color='brown')
plt.xlabel('Training Steps')
plt.ylabel('Ratio')
plt.title('Generator/Discriminator Loss Ratio')
plt.legend()
plt.grid(True)

# 绘制移动平均损失
plt.subplot(2, 2, 4)
window = 50
g_smooth = np.convolve(g_losses, np.ones(window) / window, mode='valid')
d_smooth = np.convolve(d_losses, np.ones(window) / window, mode='valid')
w_smooth = np.convolve(wasserstein_distances, np.ones(window) / window, mode='valid')
steps_smooth = steps[window - 1:]
plt.plot(steps_smooth, g_smooth, label='Generator Loss (MA)', color='blue')
plt.plot(steps_smooth, d_smooth, label='Discriminator Loss (MA)', color='red')
plt.plot(steps_smooth, w_smooth, label='Wasserstein Dist (MA)', color='green')
plt.xlabel('Training Steps')
plt.ylabel('Loss/Distance (Moving Average)')
plt.title('Smoothed Metrics (Window = {})'.format(window))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(loss_plot_dir, 'final_loss_plot.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"Final loss plots saved to {loss_plot_dir}/")

# 生成特定数字的示例图像
print("Generating example images for each digit...")
example_dir = "wgan_example_images"
os.makedirs(example_dir, exist_ok=True)

with torch.no_grad():
    for digit in range(10):
        # 为每个数字生成16个样本
        z = torch.randn(16, latent_dim)
        labels = torch.full((16,), digit, dtype=torch.long)

        if use_gpu:
            z = z.to("cuda")
            labels = labels.to("cuda")

        generated_images = generator(z, labels)
        # 将图像从[-1,1]范围转换回[0,1]范围
        generated_images = (generated_images + 1) / 2
        torchvision.utils.save_image(generated_images, os.path.join(example_dir, f"digit_{digit}.png"), nrow=4)

print(f"Example images saved to {example_dir}/")