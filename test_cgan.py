#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_gan.py
# Author            : none <none>
# Date              : 14.04.2022
# Last Modified Date: 15.04.2022
# Last Modified By  : none <none>
""" 基于MNIST 实现对抗生成网络 (GAN) """

import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt  # 新增：用于绘制损失图像

# 定义图像尺寸和潜在空间维度
image_size = [1, 28, 28]  # MNIST图像尺寸: 1通道, 28x28像素
latent_dim = 96  # 生成器输入噪声向量的维度
label_emb_dim = 32
batch_size = 64  # 训练批量大小
use_gpu = torch.cuda.is_available()  # 检查是否可用GPU
save_dir = "cgan_generated_images"
os.makedirs(save_dir, exist_ok=True)

# 创建目录用于保存损失图像
loss_plot_dir = "cgan_loss_plots"
os.makedirs(loss_plot_dir, exist_ok=True)


class Generator(nn.Module):
    """生成器网络 - 将随机噪声转换为逼真的图像"""

    def __init__(self):
        super(Generator, self).__init__()

        self.embedding = nn.Embedding(10, label_emb_dim)
        # 定义生成器的神经网络结构
        self.model = nn.Sequential(
            # 全连接层，将噪声向量映射到更高维度
            nn.Linear(latent_dim + label_emb_dim, 128),
            torch.nn.BatchNorm1d(128),  # 批标准化，加速训练并提高稳定性
            torch.nn.GELU(),  # GELU激活函数，比ReLU更平滑

            # 逐步增加网络容量
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),

            # 最终输出层，生成与图像像素数相同的输出
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            # 使用Sigmoid将输出压缩到[0,1]范围，符合图像像素值范围
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        # 输入z: [batch_size, latent_dim] 随机噪声向量

        # 通过生成器网络
        label_embedding = self.embedding(labels)
        label_embedding = abs(label_embedding * 0.1)
        z = torch.cat([z, label_embedding], dim=-1)
        output = self.model(z)
        # 将扁平化的输出重塑为图像格式 [batch_size, channels, height, width]
        image = output.reshape(z.shape[0], *image_size)

        return image


class Discriminator(nn.Module):
    """判别器网络 - 区分真实图像和生成图像"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10, label_emb_dim)
        # 定义判别器的神经网络结构
        self.model = nn.Sequential(
            # 输入层，接收扁平化的图像
            nn.Linear(np.prod(image_size, dtype=np.int32) + label_emb_dim, 512),
            torch.nn.GELU(),

            # 逐步减少网络容量
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),

            # 输出层，单一值表示图像为真的概率
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出范围[0,1]，表示概率
        )

    def forward(self, image, labels):
        # 输入图像: [batch_size, 1, 28, 28]

        # 将图像扁平化后通过判别器网络
        label_embedding = self.embedding(labels)
        label_embedding = abs(label_embedding * 0.1)
        prob = self.model(torch.cat([image.reshape(image.shape[0], -1), label_embedding], dim=-1))

        return prob  # 返回图像为真实图像的概率


# 数据准备
# 加载MNIST数据集
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),  # 调整大小
                                             torchvision.transforms.ToTensor(),  # 转换为张量
                                             # 注意：这里没有标准化，因为Sigmoid输出已在[0,1]范围内
                                         ]
                                     )
                                     )
# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 定义优化器
# 使用Adam优化器，设置不同的超参数
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

# 定义损失函数 - 二元交叉熵损失
loss_fn = nn.BCELoss()
# 创建真实和虚假标签
labels_one = torch.ones(batch_size, 1)  # 真实图像标签为1
labels_zero = torch.zeros(batch_size, 1)  # 生成图像标签为0

# 如果可用，将模型和数据移动到GPU
if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

# 新增：初始化损失记录列表
g_losses = []
d_losses = []
recons_losses = []
real_losses = []
fake_losses = []
steps = []

# 训练循环
num_epoch = 200  # 训练轮数
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, labels = mini_batch  # 获取真实图像，忽略标签

        # 生成随机噪声作为生成器输入
        z = torch.randn(batch_size, latent_dim)

        # 将数据移动到GPU（如果可用）
        if use_gpu:
            gt_images = gt_images.to("cuda")
            labels = labels.to("cuda")
            z = z.to("cuda")

        # 训练生成器
        # 1. 生成图像
        pred_images = generator(z, labels)
        # 2. 清零生成器梯度
        g_optimizer.zero_grad()

        # 计算重建损失（L1损失） - 使生成图像更接近真实图像
        recons_loss = torch.abs(pred_images - gt_images).mean()
        # 计算对抗损失 - 希望判别器将生成图像判断为真实
        g_loss = recons_loss * 0.05 + loss_fn(discriminator(pred_images, labels), labels_one)

        # 反向传播和优化
        g_loss.backward()
        g_optimizer.step()

        # 训练判别器
        # 1. 清零判别器梯度
        d_optimizer.zero_grad()

        # 计算真实图像的损失 - 希望判别器将真实图像判断为真实
        real_loss = loss_fn(discriminator(gt_images, labels), labels_one)
        # 计算生成图像的损失 - 希望判别器将生成图像判断为虚假
        # 使用detach()防止梯度传播到生成器
        fake_loss = loss_fn(discriminator(pred_images.detach(), labels), labels_zero)
        # 总判别器损失
        d_loss = (real_loss + fake_loss)

        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了

        # 反向传播和优化
        d_loss.backward()
        d_optimizer.step()

        # 记录损失值
        current_step = len(dataloader) * epoch + i
        steps.append(current_step)
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        recons_losses.append(recons_loss.item())
        real_losses.append(real_loss.item())
        fake_losses.append(fake_loss.item())

        # 定期输出训练状态
        if i % 50 == 0:
            print(
                f"step:{len(dataloader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        # 定期保存生成的图像样本
        if i % 400 == 0:
            image = pred_images[:16].data  # 取前16个生成图像
            torchvision.utils.save_image(image, os.path.join(save_dir, f"image_{len(dataloader) * epoch + i}.png"),
                                         nrow=4)


# 训练结束后绘制最终损失图像
print("Training completed! Generating final loss plots...")
plt.figure(figsize=(15, 10))

# 绘制生成器和判别器损失
plt.subplot(2, 3, 1)
plt.plot(steps, g_losses, label='Generator Loss', color='blue')
plt.plot(steps, d_losses, label='Discriminator Loss', color='red')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss')
plt.legend()
plt.grid(True)

# 绘制详细损失
plt.subplot(2, 3, 2)
plt.plot(steps, recons_losses, label='Reconstruction Loss', color='green')
plt.plot(steps, real_losses, label='Real Loss', color='orange')
plt.plot(steps, fake_losses, label='Fake Loss', color='purple')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Detailed Loss Components')
plt.legend()
plt.grid(True)

# 绘制生成器相关损失
plt.subplot(2, 3, 3)
plt.plot(steps, g_losses, label='Generator Loss', color='blue')
plt.plot(steps, recons_losses, label='Reconstruction Loss', color='green')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Generator Loss Components')
plt.legend()
plt.grid(True)

# 绘制判别器相关损失
plt.subplot(2, 3, 4)
plt.plot(steps, d_losses, label='Discriminator Loss', color='red')
plt.plot(steps, real_losses, label='Real Loss', color='orange')
plt.plot(steps, fake_losses, label='Fake Loss', color='purple')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Discriminator Loss Components')
plt.legend()
plt.grid(True)

# 绘制损失比率
plt.subplot(2, 3, 5)
ratio = [g / d if d != 0 else 0 for g, d in zip(g_losses, d_losses)]
plt.plot(steps, ratio, label='G/D Loss Ratio', color='brown')
plt.xlabel('Training Steps')
plt.ylabel('Ratio')
plt.title('Generator/Discriminator Loss Ratio')
plt.legend()
plt.grid(True)

# 绘制移动平均损失
plt.subplot(2, 3, 6)
window = 50
g_smooth = np.convolve(g_losses, np.ones(window) / window, mode='valid')
d_smooth = np.convolve(d_losses, np.ones(window) / window, mode='valid')
steps_smooth = steps[window - 1:]
plt.plot(steps_smooth, g_smooth, label='Generator Loss (MA)', color='blue')
plt.plot(steps_smooth, d_smooth, label='Discriminator Loss (MA)', color='red')
plt.xlabel('Training Steps')
plt.ylabel('Loss (Moving Average)')
plt.title('Smoothed Loss (Window = {})'.format(window))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(loss_plot_dir, 'final_loss_plot.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"Final loss plots saved to {loss_plot_dir}/")