#!/usr/bin/env python3
"""关卡 1 的预测表：按访存下界折算"索引筛选"相对"完整读取"每步能省多少字节和时间。

这是预测，不是测量。假设：batch=1、BF16、解码阶段纯访存受限、每步读一遍全部权重和全部参与注意力的 KV。
两种情景：
  S1 只计索引键读取 + 入选 KV 读取一次（gather 与注意力融合，或子集留在片上缓存）
  S2 另计 gather 的写入与注意力的重读，即入选 KV 流量按三份估（GPT 6 Pro 提出的保守情景）
eager 行：每层约 7 个额外小算子 × 层数 × 每次 10–20 µs 的 CPU 提交开销；GPU 异步执行可能覆盖其中一部分，所以这是"新增 CPU 时间"，不是墙钟增量。
用法：python3 roofline.py [--bw 716.8e9] [--rho 0.03125] [--r 32] [--fixed 68]
"""
import argparse

MODELS = {
    # name: (layers, q_heads, kv_heads, head_dim, total_params)
    "Qwen3-0.6B": (28, 16, 8, 128, 596_049_920),
    "Qwen3-1.7B": (28, 16, 8, 128, 1_720_574_976),   # 以运行时 numel() 为准
    "Qwen3-4B":   (36, 32, 8, 128, 4_022_468_096),
}
NS = (8192, 16384, 32768)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=716.8e9, help="显存带宽 B/s，默认 RTX 4080 标称")
    ap.add_argument("--rho", type=float, default=1/32, help="较早历史选取比例 k = rho*N")
    ap.add_argument("--r", type=int, default=32, help="索引向量宽度")
    ap.add_argument("--fixed", type=int, default=68, help="固定保留位置数（首 4 + 近 64）")
    ap.add_argument("--bytes", type=int, default=2, help="每个数的字节数，BF16=2")
    ap.add_argument("--launch_us", type=float, default=15.0, help="eager 每个算子的 CPU 提交开销 µs")
    ap.add_argument("--extra_ops", type=int, default=7, help="D 每层新增小算子数")
    a = ap.parse_args()
    b = a.bytes
    print(f"假设：bw={a.bw/1e9:.1f} GB/s, rho=1/{round(1/a.rho)}, r={a.r}, fixed={a.fixed}, BF16, batch=1\n")
    print("| 模型 | N | 权重 GB | 完整KV GB | 注意力占每步流量 | k | D读取 S1 GB | 省 S1 ms | 省 S1 % | D读取 S2 GB | 省 S2 ms | 省 S2 % | 理想步长 ms | eager新增CPU ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, (L, Hq, G, d, P) in MODELS.items():
        W = P * b
        extra_cpu = a.extra_ops * L * a.launch_us * 1e-6
        for N in NS:
            k = round(a.rho * N)
            KV = 2 * L * N * G * d * b
            idx = L * N * G * a.r * b
            sel = 2 * L * (k + a.fixed) * G * d * b
            step = (W + KV) / a.bw
            d1 = idx + sel; d2 = idx + 3 * sel
            s1 = KV - d1; s2 = KV - d2
            print(f"| {name} | {N} | {W/1e9:.2f} | {KV/1e9:.2f} | {100*KV/(W+KV):.0f}% | {k} | {d1/1e9:.2f} | {1e3*s1/a.bw:.2f} | {100*s1/(W+KV):.0f}% | {d2/1e9:.2f} | {1e3*s2/a.bw:.2f} | {100*s2/(W+KV):.0f}% | {1e3*step:.1f} | {1e3*extra_cpu:.1f} |")
    print("\n索引键读取是 D 的流量下限：r/(2d) = %.3f，即无论 k 多小，D 至少读完整 KV 的 %.1f%%。" % (a.r/(2*128), 100*a.r/(2*128)))
    L, Hq, G, d, P = MODELS["Qwen3-4B"]
    for N in (16384, 32768):
        tot = P*b + 2*L*N*G*d*b + L*N*G*a.r*b
        print(f"4B@{N}: 权重+KV+索引缓存 = {tot/1e9:.2f} GB（不含激活、临时量、分配器；16 GB 卡是否放得下由峰值预检决定）")

if __name__ == "__main__":
    main()
