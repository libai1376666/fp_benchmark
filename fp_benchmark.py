#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
浮点算力测试单元
"""

import time
import argparse
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os

import sys
import subprocess

# 尝试导入 tqdm，如果失败则通过 pip 动态安装
try:
    from tqdm import trange
except ImportError:
    print("tqdm 未安装，正在自动安装…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import trange


class MatMulBenchmark(ABC):
    """抽象基类：N×N 矩阵乘法基准测试。"""

    def __init__(self, size: int, dtype: torch.dtype, iterations: int):
        """
        Args:
            size: 矩阵维度 N
            dtype: torch.dtype，数据类型
            iterations: 测试迭代次数
        """
        self.size = size
        self.dtype = dtype
        self.iterations = iterations        # 测试多次, 以计算平均值
        self.a = None  # 左矩阵
        self.b = None  # 右矩阵

    @abstractmethod
    def prepare(self):
        """为 benchmark 初始化张量并做预热。"""
        pass

    @abstractmethod
    def run(self) -> float:
        """
        执行 iterations 次矩阵乘法，返回总耗时（秒）。
        注意：不做任何 compute→time 的转换，这里直接返回秒数。
        """
        pass

    def compute_tflops(self, elapsed_s: float) -> float:
        """
        根据 elapsed_s 计算 TFLOPS。
        一次 N×N 矩阵乘法的浮点运算量 ≈ 2*N^3
        """
        n = self.size
        total_flops = 2 * (n ** 3) * self.iterations
        return total_flops / elapsed_s / 1e12

    def benchmark(self):
        """
        执行完整的基准测试：prepare → run → compute_tflops。
        返回 (latency_ms, tflops)。
        """
        self.prepare()
        elapsed_s = self.run()
        tflops = self.compute_tflops(elapsed_s)
        return elapsed_s * 1000, tflops


class CPUBenchmark(MatMulBenchmark):
    """CPU 路径的矩阵乘法基准。"""

    def __init__(self, size: int, dtype: torch.dtype, iterations: int, threads=None):
        super().__init__(size, dtype, iterations)
        # 设置 BLAS/OpenMP 线程数, 从而利用指定CPU算力
        n_threads = threads if threads else os.cpu_count()
        torch.set_num_threads(n_threads)
        # set_num_interop_threads 只能调用一次（或在并行工作前），这里捕获二次调用的错误
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            # 已经设置过或已开始并行，就忽略
            pass

    def prepare(self):
        # 在 CPU 上构造随机张量并预热
        self.a = torch.randn(self.size, self.size, dtype=self.dtype, device='cpu')
        self.b = torch.randn(self.size, self.size, dtype=self.dtype, device='cpu')
        # 再次确认线程数已生效
        print(f"{self.dtype}: Using {torch.get_num_threads()} threads for CPU benchmark")

        # 分配输出缓冲区，和 a/b 一样大小
        self.res = torch.empty_like(self.a)

        # 预热：torch.no_grad() + 重用 out，逼热 BLAS 缓存
        with torch.no_grad():
            for _ in range(5):
                torch.matmul(self.a, self.b, out=self.res)

    def run(self) -> float:
        # 使用高精度计时器测量多次迭代
        start = time.perf_counter()
        with torch.no_grad():
            # 用 trange 包裹，显示进度
            for _ in trange(self.iterations, desc="CPU MatMul"):
                # 每次都往同一块内存写，去掉分配新张量的开销
                torch.matmul(self.a, self.b, out=self.res)
        end = time.perf_counter()
        return end - start


class GPUBenchmark(MatMulBenchmark):
    """CUDA 单卡路径的矩阵乘法基准。"""

    def __init__(self, size, dtype, iterations, gpu_id=0, use_tf32=True):
        super().__init__(size, dtype, iterations)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")

        # 1. 设置当前进程默认使用哪张 GPU
        torch.cuda.set_device(gpu_id)

        # 2. 显示指明当前运行GPU, 通过和默认指定配合, 确保当前调用跑在同一张GPU上
        self.device = torch.device(f'cuda:{gpu_id}')

        self.use_tf32 = use_tf32
        # 创建单次复用的 CUDA 事件
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt   = torch.cuda.Event(enable_timing=True)
        # 可选：创建一个自定义 Stream
        self.stream = torch.cuda.Stream()

    def prepare(self):
        # 切换 TF32 模式（仅对 FP32 生效）
        if self.use_tf32 and self.dtype is torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # 在 GPU 上构造随机张量并预热
        self.a = torch.randn(self.size, self.size, dtype=self.dtype, device=self.device)
        self.b = torch.randn(self.size, self.size, dtype=self.dtype, device=self.device)
        print(f"{self.dtype}: prepared on {self.device}, TF32={'ON' if self.use_tf32 else 'OFF'}")
        for _ in range(5):
            _ = self.a @ self.b
        torch.cuda.synchronize()

    def run(self) -> float:
        # 使用 no_grad, 自定义 Stream 以及预分配好的 Events
        torch.cuda.synchronize(self.device)
        with torch.no_grad(), torch.cuda.stream(self.stream):
            # 记录开始
            self.start_evt.record(self.stream)
            for _ in range(self.iterations):
                # 只做前向，避免 Python 层进度条干扰，这里不放 tqdm
                self.a.mm(self.b)
            # 记录结束
            self.end_evt.record(self.stream)

        # 同步流并测时
        self.end_evt.synchronize()
        elapsed_ms = self.start_evt.elapsed_time(self.end_evt)
        return elapsed_ms / 1000.0


# 定义简易 MatMul 模型, 用来给DDP模式进行计算
class SimpleMatMul(torch.nn.Module):
    def __init__(self, size, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(size, size, dtype=dtype)
        )
    def forward(self, x):
        return x @ self.weight

class DDPBenchmark(MatMulBenchmark):
    """多卡 DDP 基准测试，汇总各卡 TFLOPS。"""

    def __init__(self, size, dtype, iterations, gpu_ids, use_tf32=True):
        """
        Args:
          size: 矩阵维度
          dtype: torch.dtype
          iterations: 迭代次数
          gpu_ids: GPU 索引列表
          use_tf32: FP32 时是否启用 TF32
        """
        super().__init__(size, dtype, iterations)
        self.gpu_ids = gpu_ids
        self.world_size = len(gpu_ids)
        self.use_tf32 = use_tf32

    def _ddp_worker(self, rank, return_dict):
        '''
        ddp工作模块
        '''
        # 通信环境
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        # 内部启动 NCCL 通道，完成所有进程之间的握手
        dist.init_process_group('nccl', rank=rank, world_size=self.world_size)      # 后端 用 nccl（针对多 GPU 同机或跨机通信的高效库）, rank：这是当前进程在整个训练作业中的编号（从 0 到 world_size–1）, world_size：一共要跑多少个进程（等同于要用多少 GPU）
        # 绑定卡
        torch.cuda.set_device(self.gpu_ids[rank])
        device = torch.device(f'cuda:{self.gpu_ids[rank]}')

        # 初始化模型 & DDP 包装
        model = SimpleMatMul(self.size, self.dtype).to(device)
        if self.dtype is torch.float32 and self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        ddp_model = DDP(model, device_ids=[self.gpu_ids[rank]])

        # 构造输入 & warm-up
        x = torch.randn(self.size, self.size, dtype=self.dtype, device=device)
        for _ in range(5):
            _ = ddp_model(x)
        torch.cuda.synchronize(device)

        # 计时
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_evt.record()
            for _ in range(self.iterations):
                _ = ddp_model(x)
            end_evt.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_evt.elapsed_time(end_evt)

        # 计算本卡计算时间, 并添加到列表
        elapsed_s = elapsed_ms / 1000.0
        return_dict[rank] = elapsed_s

        # 关闭并销毁当前进程所加入的分布式通信组, 释放资源
        dist.destroy_process_group()

    def prepare(self):
        # DDPBenchmark 不在主进程做准备，全部在 worker 中
        print(f"{self.dtype}: prepared on ddp mode(cuda: {self.gpu_ids}), TF32={'ON' if self.use_tf32 else 'OFF'}")
        pass

    def run(self) -> float:
        # 使用 Manager 字典收集各卡结果
        manager = mp.Manager()
        return_dict = manager.dict()

        # spawn DDP worker 进程
        mp.spawn(
            self._ddp_worker,
            args=(return_dict,),
            nprocs=self.world_size,
            join=True
        )

        # 并行计算总耗时= 各卡耗时的最大值
        max_elapsed_s = max(return_dict.values())
        return max_elapsed_s

    def compute_tflops(self, elapsed_s: float) -> float:
        """
        根据 max_elapsed_s 计算 TFLOPS。
        DDP模式中, 计算量为各个卡计算量之和
        """
        n = self.size
        total_flops = 2 * (n ** 3) * self.iterations * self.world_size
        return total_flops / elapsed_s / 1e12
def main():
    parser = argparse.ArgumentParser(description="Matrix Multiply Benchmark (OOP)")
    parser.add_argument('--size',  type=int,   default=1024 * 8,
                        help='矩阵维度 N (测试 N×N)')
    parser.add_argument('--iters', type=int,   default=100,
                        help='迭代次数')
    parser.add_argument('--dtypes', nargs='+',
                        default=['fp32', 'fp16'],
                        choices=['fp32', 'fp64', 'fp16', 'bf16'],
                        help='要测试的数据类型')
    parser.add_argument('--device', type=str, default='cuda',
                        help='测试设备：cpu, cuda, ...')
    parser.add_argument('--gpu_ids',   type=str, default='0',
                        help='调用的GPU id 列表，逗号分隔，如 "0" 或 "0,1,2", 当填入1个时, 表示测试单卡性能')
    parser.add_argument('--use_tf32', type=bool, default=True,
                        help='是否使用Tensor Core来加速fp32计算, 仅在Ampere及之后架构才支持')
    parser.add_argument('--threads', type=int, default=None,
                        help='设置CPU线程数, 不指定时为所有线程')
    args = parser.parse_args()

    # dtype 名称到 torch.dtype 的映射
    dtype_map = {
        'fp32': torch.float32,
        'fp64': torch.float64,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }

    # 根据 --device 决定要跑哪些基准
    devices = args.device

    print(f"Benchmarking {args.size}×{args.size}, iters={args.iters}\n")

    if devices == 'cpu':
        # 遍历数据类型
        for name in args.dtypes:
            # CPU 仅支持 fp32, fp64
            if name not in ('fp32', 'fp64'):
                print(f"Skipping CPU-{name}: not supported")
                continue

            dtype = dtype_map[name]
            key = f"{devices}-{name}"
            try:
                bench = CPUBenchmark(args.size, dtype, args.iters, args.threads)
                ms, tflops = bench.benchmark()
                print(f"{key}: {ms:.2f} ms, {tflops:.2f} TFLOPS")
            except Exception as e:
                print(f"{key}: Error - {e}")
    elif devices == 'cuda':
        # 遍历数据类型
        for name in args.dtypes:
            # BF16 在 GPU 上需硬件 & PyTorch 支持
            if name == 'bf16' and not torch.cuda.is_bf16_supported():
                print("Skipping GPU-bf16: hardware/PyTorch unsupported")
                continue
            # 解析 GPU 列表
            gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip().isdigit()]
            dtype = dtype_map[name]

            # 单卡模式
            if len(gpu_ids) == 1:
                    key = f"{devices}:{gpu_ids[0]}-{name}"
                    try:
                        bench = GPUBenchmark(args.size, dtype, args.iters, gpu_ids[0], args.use_tf32)
                        ms, tflops = bench.benchmark()
                        print(f"{key}: {ms:.2f} ms, {tflops:.2f} TFLOPS")
                    except Exception as e:
                        print(f"{key}: Error - {e}")
            # 多卡模式
            else:
                key = f"DDP Model:{devices}:{gpu_ids}-{name}"
                try:
                    bench = DDPBenchmark(args.size, dtype, args.iters, gpu_ids, args.use_tf32)
                    ms, tflops = bench.benchmark()
                    print(f"{key}: {ms:.2f} ms, {tflops:.2f} TFLOPS")
                except Exception as e:
                    print(f"{key}: Error - {e}")
    else:
        print(f'{devices} is not supported')


if __name__ == '__main__':
    main()
