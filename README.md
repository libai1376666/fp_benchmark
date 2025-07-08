# 计算设备浮点算力

快速测试芯片浮点算力，目前仅支持CPU，Nvidia GPU

# 快速使用

1. 安装环境和依赖:

```sh
# 创建conda环境
conda create --name fp_benchmark python=3.11

# 激活环境
conda activate fp_benchmark

# 安装pytorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

2. 快速运行

```sh
python fp_benchmark --device cuda --gpu_ids 0
```

# 参数介绍

```py
add_argument('--size',  type=int,   default=1024 * 8,
                    help='矩阵维度 N (测试 N×N)')
add_argument('--iters', type=int,   default=100,
                    help='迭代次数')
add_argument('--dtypes', nargs='+',
                    default=['fp32', 'fp16'],
                    choices=['fp32', 'fp64', 'fp16', 'bf16'],
                    help='要测试的数据类型')
add_argument('--device', type=str, default='cuda',
                    help='测试设备：cpu, cuda, ...')
add_argument('--gpu_ids',   type=str, default='0',
                    help='调用的GPU id 列表，逗号分隔，如 "0" 或 "0,1,2", 当填入1个时, 表示测试单卡性能')
add_argument('--use_tf32', type=bool, default=True,
                    help='是否使用Tensor Core来加速fp32计算, 仅在Ampere及之后架构才支持')
add_argument('--threads', type=int, default=None,
                    help='设置CPU线程数, 不指定时为所有线程')
```

> 注: 测试CPU时, 矩阵维度建议设小点, 不然运行时间非常长


# 测试结果收集

对于不支持Tensor Core加速的设备, 单精度为FP32, 否则为TF32

单位统一为`TFLOPS`

| Device   | FP32/TF32 | FP16   |
| ------ | ---- | ------ |
| Nvidia RTX 4070Ti Super  | 46.63   | 92.34 |
| Nvidia RTX 3060 12G     | 13.13   | 25.84 |
| Nvidia RTX 4090 24G     | 88.20   | 160.52 |
| AMD Ryzen 5900X         | 1.07    | N/A |