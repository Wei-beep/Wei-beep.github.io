---
layout: default
title: RV1106 RKNN 双输入模型 INT8 量化部署踩坑记
---

# RV1106 RKNN 双输入模型 INT8 量化部署踩坑记

> 作者：Wei-beep | 日期：2026-10
> 硬件：瑞芯微 RV1106（ARM Cortex-A7 @1.2GHz，RKNN NPU 0.5TOPS，64MB DDR2）
> 软件：Buildroot Linux（uclibc）+ librknnmrt 1.6.0 + arm-rockchip830-linux-uclibcgnueabihf-gcc

## 背景

我在做晶圆缺陷检测的毕设项目，模型是一个双输入 CNN：图像分支（64×64×3 uint8 RGB）+ 特征分支（16维 float32 统计特征），融合后输出 8 类缺陷分类。模型训练用 PyTorch，导出 ONNX 后用 RKNN Toolkit 做 INT8 量化，部署到 RV1106 的 NPU 上推理。

整个过程踩了 5 个坑，每一个都花了不少时间定位。写下来希望帮到后面的人。

---

## 坑1：图像分支手动 INT8 量化，精度从 92% 暴跌到 30%

### 现象

板端推理准确率从 PC 端的 ~90% 直接掉到 30% 以下，几乎等于随机猜。

### 定位过程

1. 先怀疑模型转换有问题，重新量化了三次，结果一样。
2. 打印 NPU 输入数据，发现 image 分支喂进去的值在 0~127 范围（我手动做了 int8 量化）。
3. 回头看 RKNN 量化时的配置：`mean_values=[[0,0,0]], std_values=[[255,255,255]]`。
4. 这意味着 RKNN 在转换时已经把归一化（除以 255）融合进了模型的第一层，量化参数 image scale=0.003922（=1/255），zero_point=-128。
5. NPU 内部会自动做 `uint8_raw * scale + zp` 的反归一化，我手动再做一次 int8 量化等于**双重量化**。

### 解决方案

- **image 分支**：直接喂 `uint8` 原始 RGB 像素（0~255），**不要手动量化**。
- **features 分支**：这个分支没有在 RKNN config 里设置 mean/std，所以 NPU 不会自动归一化，**必须手动 int8 量化**：

```c
// features 量化参数：scale=0.099453, zero_point=13
int8_t feat_q[16];
for (int i = 0; i < 16; i++) {
    int q = (int)roundf(feat_float[i] / 0.099453f + 13.0f);
    feat_q[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}
```

### 教训

**不是所有输入都要手动量化。** 用 RKNN config 里的 `mean_values/std_values` 做了归一化的输入，NPU 内部已融合，喂原始数据；没做的输入才需要手动量化。判断依据就是看每个输入的量化参数——如果 scale 是 1/255 这种归一化值，说明融合了。

---

## 坑2：rknn_inputs_set 轻量 API 恒返回 -5

### 现象

使用 `rknn_inputs_set(ctx, n_inputs, inputs)` 的轻量调用方式（直接传指针数组），始终返回错误码 -5，模型无法推理。

### 定位过程

1. 查 RKNN API 文档，错误码 -5 对应 `RKNN_ERR_PARAM_INVALID`（参数无效）。
2. 检查输入数量、索引、buffer 大小，全部正确。
3. 单输入模型用同样的方式没问题，双输入就报错。
4. 切换到标准 `rknn_input` 结构体方式，逐个设置 `index/type/buf/size/pass_through`，问题消失。

### 解决方案

使用标准 API 流程，不要用轻量方式：

```c
rknn_input inputs[2];
memset(inputs, 0, sizeof(inputs));

// image 输入：uint8 原始像素
inputs[0].index = 0;
inputs[0].type = RKNN_TENSOR_UINT8;
inputs[0].size = 64 * 64 * 3;
inputs[0].buf = img_buf;
inputs[0].pass_through = 0;

// features 输入：手动量化后的 int8
inputs[1].index = 1;
inputs[1].type = RKNN_TENSOR_INT8;
inputs[1].size = 16;
inputs[1].buf = feat_q;
inputs[1].pass_through = 0;

int ret = rknn_inputs_set(ctx, 2, inputs);
if (ret < 0) {
    printf("rknn_inputs_set failed: %d\n", ret);
    return -1;
}
```

### 教训

RKNN 轻量 API 对双输入（多输入）模型支持有 bug，**双输入模型一律用标准 `rknn_input` 结构体**。这个问题在 rknn-toolkit2 的 issue 区有人提过，但官方文档没有明确警告。

---

## 坑3：零拷贝输出改 float32 被 NPU 拒绝

### 现象

想跳过输出反量化，直接把输出张量类型设为 `RKNN_TENSOR_FLOAT32` 让 NPU 输出 float，NPU 直接报错。

### 根因

INT8 量化模型的输出张量类型在**模型转换时就固定了**，运行时不可更改。`rknn_output` 的 `is_prealloc` 和 `type` 字段只能在模型支持的类型范围内选择，量化模型的输出就是 int8，不能改成 float32。

### 解决方案

获取 int8 输出后手动反量化：

```c
rknn_output outputs[1];
memset(outputs, 0, sizeof(outputs));
outputs[0].want_float = 0;  // 不要让SDK反量化，自己来
rknn_outputs_get(ctx, 1, outputs, NULL);

int8_t *out_int8 = (int8_t *)outputs[0].buf;
// output 量化参数：scale=0.255337, zero_point=3
float logits[8];
for (int i = 0; i < 8; i++) {
    logits[i] = ((float)out_int8[i] - 3.0f) * 0.255337f;
}

rknn_outputs_release(ctx, 1, outputs);
```

> 也可以设 `want_float = 1` 让 RKNN SDK 自动反量化，但手动反量化可以确认 scale/zp 是否正确，调试阶段建议手动。

---

## 坑4：glibc 交叉编译器导致 ABI 段错误

### 现象

用 `arm-linux-gnueabihf-gcc`（glibc 版）编译的程序，scp 到板子上运行，直接段错误（Segmentation fault），连 main 都进不去。

### 定位过程

1. 用 `file wafer_direct` 看编译产物，显示 `ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), dynamically linked`。
2. 板子上用 `readelf -d` 看依赖，发现依赖 `ld-linux-armhf.so.3`，而板子上只有 `ld-uClibc.so.1`。
3. RV1106 的 Buildroot 用的是 **uclibc** 不是 glibc，两套 C 库的 ABI 不兼容。

### 解决方案

必须使用瑞芯微官方的 uclibc 交叉编译器：

```bash
# 错误：glibc 版，会段错误
arm-linux-gnueabihf-gcc wafer_direct.c -o wafer_direct ...

# 正确：uclibc 版
arm-rockchip830-linux-uclibcgnueabihf-gcc wafer_direct.c -o wafer_direct \
  -I./ \
  -L$HOME/.local/lib/python3.8/site-packages/rknn/3rdparty/\
platform-tools/runtime/Linux/librknn_api/armhf-uclibc \
  -lrknnmrt -lm -lpthread -O2
```

同时链接的 `librknnmrt.so` 也必须是 **armhf-uclibc** 版本，不能用 armhf 版（glibc）。

---

## 坑5：E2E 端到端时延显示 773ms，实际只有 2ms

### 现象

网页看板显示平均端到端时延 773ms，首帧时延 37818ms，看起来系统极慢。但 NPU 推理只要 2ms，矛盾。

### 定位过程

1. 检查计时代码，发现 E2E 计时起点设在 `recv()` 调用之前。
2. `recv()` 是阻塞调用，在没有数据到达时会一直等待，这段空闲时间被算进了 E2E 时延。
3. 首帧 37818ms 是因为板子启动后等了 37 秒才收到第一个推理请求。

### 解决方案

计时起点改为**全量收包完成之后**：

```c
// 错误：计时包含了 recv 等待空闲时间
struct timeval t_start, t_end;
gettimeofday(&t_start, NULL);
int n = recv_n(client_fd, buf, total_len);  // 阻塞等待
gettimeofday(&t_end, NULL);
// e2e_ms = (t_end - t_start)  ← 包含了等待时间

// 正确：收包完成后才开始计时
int n = recv_n(client_fd, buf, total_len);
gettimeofday(&t_start, NULL);  // 收包完成，开始推理计时
do_infer(buf, &result);
gettimeofday(&t_end, NULL);
// e2e_ms = (t_end - t_start)  ← 纯推理+处理时间
```

修正后 E2E 时延从 773ms 降到 ~2-3ms。

### 教训

**性能数据的计时起点决定数据的含义。** 做性能测试时必须明确计时边界，否则会得出完全错误的结论。

---

## 最终性能

| 指标 | 值 |
|---|---|
| 板端准确率（200张独立子集） | 92.5%（185/200） |
| NPU 单帧推理 | 2.12 ms |
| CPU 推理（对比） | 17.4 ms |
| NPU 加速比 | 8.3x |
| 内存占用 | < 10 MB（64MB 总内存） |

---

## 总结：RKNN 双输入模型部署 Checklist

- [ ] 确认每个输入的量化参数，融合了归一化的喂原始数据，没融合的手动量化
- [ ] 双输入模型用标准 `rknn_input` 结构体 API，不用轻量调用
- [ ] 输出类型在转换时固定，int8 输出手动反量化
- [ ] 交叉编译器和 librknnmrt 都必须是 uclibc 版
- [ ] `LD_LIBRARY_PATH` 包含 `/oem/usr/lib`
- [ ] 性能计时起点设在收包完成后，不包含空闲等待
- [ ] 二进制加执行权限，开机自启脚本里也要 chmod

---

## 参考

- [rknn-toolkit2 GitHub](https://github.com/rockchip-linux/rknn-toolkit2)
- [rknpu2 GitHub（含 RV1106 示例）](https://github.com/rockchip-linux/rknpu2)
- [RKNN API 文档](https://github.com/rockchip-linux/rknpu2/tree/master/doc)
- 本项目完整代码：[rv1106-wafer-defect-inspection](https://github.com/Wei-beep/rv1106-wafer-defect-inspection)
