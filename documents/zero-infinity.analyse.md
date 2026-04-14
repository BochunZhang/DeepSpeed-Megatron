# DeepSpeed ZeRO-Infinity (Stage 3) 详细分析

## 目录
1. [配置概述](#配置概述)
2. [数据结构](#数据结构)
3. [Forward 阶段流程](#forward-阶段流程)
4. [Backward 阶段流程](#backward-阶段流程)
5. [Optimizer Step 阶段流程](#optimizer-step-阶段流程)
6. [通信操作汇总](#通信操作汇总)
7. [Micro-Batch 处理差异 (GAS=4)](#micro-batch-处理差异-gas4)

---

## 配置概述

ZeRO-Infinity 是 DeepSpeed ZeRO Stage 3 的增强版本，支持将优化器状态（参数、梯度、momentum、variance）卸载到 CPU 或 NVMe 存储。

### 关键配置项

#### 1. 参数卸载配置 (`zero_offload_param`)

```python
zero_offload_param = {
    "device": "none" | "cpu" | "nvme",  # 卸载设备
    "nvme_path": "/path/to/nvme",        # NVMe 设备路径
    "buffer_count": 5,                       # NVMe 缓冲池数量
    "buffer_size": 1e8,                      # NVMe 缓冲池大小
    "max_in_cpu": 1e9,                     # 保持在 CPU 中的参数元素数
    "pin_memory": False                      # 使用 page-locked 内存
}
```

**Tensor 存储位置：**
- `device: "none"` → 参数分区存储在 GPU 上
- `device: "cpu"` → 参数分区存储在 CPU 上
- `device: "nvme"` → 参数分区存储在 NVMe 上，部分可缓存在 CPU

#### 2. 优化器卸载配置 (`zero_offload_optimizer`)

```python
zero_offload_optimizer = {
    "device": "none" | "cpu" | "nvme",  # 卸载设备
    "nvme_path": "/path/to/nvme",        # NVMe 设备路径
    "buffer_count": 4,                       # NVMe 缓冲池数量
    "pin_memory": False,                     # 使用 page-locked 内存
    "pipeline_read": False,                   # 读取流水线（ZeRO-Infinity）
    "pipeline_write": False,                  # 写入流水线（ZeRO-Infinity）
    "fast_init": False,                      # 快速初始化
    "ratio": 1.0,                           # 部分卸载比例（0-1）
    "super_offload": False,                  # Superchips 高性能卸载
    "cpuadam_cores_perc": 0.8               # CPU Adam 核心百分比
}
```

**Tensor 存储位置：**
- `device: "cpu"` → FP32 参数、梯度、momentum、variance 存储在 CPU
- `device: "nvme"` → FP32 参数、梯度、momentum、variance 存储在 NVMe，通过分片处理

#### 3. 其他重要配置

```python
{
    "reduce_bucket_size": 500000000,          # 梯度聚合 bucket 大小
    "prefetch_bucket_size": 50000000,         # 参数预取 bucket 大小
    "overlap_comm": True,                     # 通信与计算重叠
    "reduce_scatter": True,                   # 使用 reduce_scatter
    "communication_data_type": torch.float16,    # 通信数据类型
    "gradient_accumulation_dtype": torch.float32  # 梯度累积数据类型
}
```

---

## 数据结构

### 核心数据结构

#### 1. 参数相关数据结构

| 数据结构 | 描述 | 存储位置 |
|----------|------|----------|
| `fp16_groups` | 按模块分组的 FP16 参数列表 | GPU (Forward 时) / 分区存储 (Backward 后) |
| `fp16_partitioned_groups` | 参数分区副本（每个 rank 持有一部分） | GPU (未卸载) / CPU / NVMe |
| `fp16_partitioned_groups_flat` | 扁平化的 FP16 参数分区 | GPU (未卸载) / CPU / NVMe |
| `fp32_partitioned_groups_flat` | 扁平化的 FP32 参数分区（优化器参数） | GPU (未卸载) / CPU / NVMe |
| `grad_partitions_flat_buffer` | 扁平化的梯度分区缓冲区 | GPU (未卸载) / CPU (Pinned) |

#### 2. 梯度相关数据结构

| 数据结构 | 描述 | 存储位置 |
|----------|------|----------|
| `averaged_gradients` | 平均后的梯度（字典：group_id → [grads]） | GPU (未卸载) / fp32_partitioned_groups_flat.grad |
| `ipg_buckets` | 独立参数梯度聚合 bucket（按 dtype 分类）| GPU |

#### 3. 优化器状态

| 数据结构 | 描述 | 存储位置 |
|----------|------|----------|
| `optimizer.state[fp32_param]` | 优化器状态（momentum、variance）| GPU (未卸载) / CPU / NVMe |

#### 4. NVTX 标记

| 标记名称 | 位置 | 阶段 |
|----------|------|------|
| `pre_sub_module_forward_function` | parameter_offload.py:465 | Forward |
| `post_sub_module_forward_function` | parameter_offload.py:484 | Forward |
| `pre_sub_module_backward_function` | parameter_offload.py:394 | Backward |
| `post_sub_module_backward_function` | parameter_offload.py:430 | Backward |
| `reduce_partition_and_remove_grads` | stage3.py:1327 | Backward |
| `__add_grad_to_ipg_bucket` | stage3.py:1406 | Backward |
| `__reduce_and_partition_ipg_grads` | stage3.py:1431 | Backward |
| `step` | stage3.py:2444 | Optimizer Step |
| `_prepare_sub_group` | stage3.py:2260 | Optimizer Step |
| `_optimizer_step` | stage3.py:1103 | Optimizer Step |
| `_reassign_or_swap_out_partitioned_parameters` | stage3.py:2424 | Optimizer Step |
| `_release_sub_group` | stage3.py:2288 | Optimizer Step |

---

## Forward 阶段流程

### 流程概览

```
开始 Forward
  ↓
pre_sub_module_forward_function (NVTX 标记)
  ↓
  1. trace_prologue - 记录参数使用轨迹
  2. fetch_sub_module - 获取当前子模块所需参数
     ↓
     2.1 _ensure_availability_of_partitioned_params - 确保 NVMe 参数可用
         - 如果参数在 NVMe：swap_in (DtoH 通信)
         - 如果参数在 CPU：HtoD 通信
     2.2 _all_gather_params - 发起 All-Gather 通信
         - GPU 集合通信：dist.all_gather 或 dist.all_reduce
     2.3 等待 All-Gather 完成
  ↓
执行前向计算
  ↓
post_sub_module_forward_function (NVTX 标记)
  ↓
  release_sub_module - 释放当前子模块参数
     ↓
     - param.partition() - 将参数返回到分区状态
     - 参数状态：AVAILABLE → NOT_AVAILABLE
     - 如果配置了参数卸载到 NVMe：swap_out
  ↓
继续下一个子模块
  ↓
Forward 完成
```

### 详细步骤分析

#### Step 1: 参数获取 (fetch_sub_module)

**位置：** `partitioned_param_coordinator.py:295`

**操作流程：**

1. **确定需要获取的参数**
   ```python
   params_to_fetch = set(iter_params(current_submodule, recurse=is_leaf))
   ```

2. **检查参数状态**
   - `param.ds_status == NOT_AVAILABLE` → 需要从分区存储获取
   - `param.ds_status == INFLIGHT` → 等待 All-Gather 完成
   - `param.ds_status == AVAILABLE` → 参数已在 GPU 上

3. **发起 All-Gather 通信**
   ```python
   # partitioned_param_coordinator.py:509-560
   def __all_gather_params_(self, params, forward, quantize):
       for param in params:
           if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
               partitioned_params.append(param)
       
       if partitioned_params:
           handle = param_group[0].all_gather_coalesced(param_group)
           for param in param_group:
               self.__inflight_param_registry[param] = handle
   ```

4. **All-Gather 具体实现**
   ```python
   # partition_parameters.py:1227-1380
   def _all_gather_sequential(params, world_size, use_secondary_tensor, ds_process_group, quantize):
       for param in params:
           # 从分区存储获取参数
           param_ds_tensor = param.ds_secondary_tensor if use_secondary_tensor else param.ds_tensor
           
           # 分配 All-Gather 缓冲区
           param_buffer = torch.empty(buffer_size, dtype=..., device="cuda")
           
           # 发起 All-Gather
           handle = dist.all_gather(param_ds_tensor.to("cuda"), param_buffer, ds_process_group)
           
           # 调整参数形状
           param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape)
   ```

**Tensor 存储位置：**
- **分区参数 (`param.ds_tensor`)**：
  - 未卸载：GPU
  - CPU 卸载：CPU (通过 `to("cuda")` 移到 GPU)
  - NVMe 卸载：NVMe (通过 swap_in 先移到 CPU，再移到 GPU)
- **完整参数 (`param.data`)**：GPU (All-Gather 结果)
- **All-Gather 缓冲区**：GPU

**通信操作：**
- **GPU-GPU 集合通信**：
  - `dist.all_gather()` - 所有 rank 的参数分区 → 每个 rank 获得完整参数
  - `dist.all_reduce()` - 所有 rank 的参数分区求和 → 每个 rank 获得完整参数
- **DtoH 通信**（NVMe 卸载）：从 NVMe 读取到 CPU
- **HtoD 通信**（CPU/NVMe 卸载）：从 CPU 复制到 GPU

**NVTX 标记：**
- `FORWARD_FETCH_SUBMIT` - 发起参数获取
- `FORWARD_FETCH_WAIT` - 等待参数获取
- `FORWARD_ALL_GATHER` - All-Gather 操作
- `FORWARD_PREFETCH_SUBMIT` - 发起参数预取

#### Step 2: 参数释放 (release_sub_module)

**位置：** `partitioned_param_coordinator.py:470`

**操作流程：**

```python
# partitioned_param_coordinator.py:564-589
def __release_param(self, param, free_data):
    if param.ds_status == ZeroParamStatus.AVAILABLE:
        param.partition(free_data=free_data)

# partition_parameters.py:1657-1700
def _partition_param(self, param, buffer=None, has_been_updated=False, free_data=True):
    if param.ds_status == ZeroParamStatus.AVAILABLE:
        if free_data:
            free_param(param)  # 释放 GPU 内存
        
        if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
            # 释放 NVMe 缓冲区
            param.nvme_swapper.remove_partition_and_release_buffers([param])
        
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE
```

**Tensor 释放：**
- `param.data` (GPU) → 设置为空张量或释放内存
- `param.ds_tensor` 保留在分区存储（GPU/CPU/NVMe）

**NVTX 标记：**
- `partition_param` - 参数分区操作

#### Step 3: 预取机制

**位置：** `partitioned_param_coordinator.py:396-464`

**操作流程：**

1. **根据轨迹确定预取参数**
   ```python
   # 从参数获取队列中取出下一个需要使用的参数
   while self.__param_queue and numel_prefetching < max_params_to_prefetch:
       param_in_trace = self.__param_queue.popleft()
       if param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
           params_to_prefetch.add(param_in_trace.param)
   ```

2. **发起异步 All-Gather**
   ```python
   self.__all_gather_params(params_to_prefetch, forward)
   ```

3. **NVMe 预取**（如果启用）
   ```python
   if self.__prefetch_nvme:
       self.__prefetch_nvme_param_partitions()
   ```

**NVTX 标记：**
- `FORWARD_PREFETCH_SUBMIT` - 发起预取

---

## Backward 阶段流程

### 流程概览

```
开始 Backward
  ↓
backward_prologue (NVTX 标记)
  ↓
  - 如果启用 swap_optimizer：optimizer_swapper.pre_backward()
  ↓
PyTorch 反向传播
  ↓
每个参数梯度计算完成后触发 backward hook:
  ↓
reduce_partition_and_remove_grads (NVTX 标记)
  ↓
  1. reduce_independent_p_g_buckets_and_remove_grads
     ↓
     1.1 检查 IPG bucket 是否满
     1.2 将梯度添加到 IPG bucket：__add_grad_to_ipg_bucket
         - 将梯度复制到连续缓冲区（GPU）
     1.3 如果 bucket 满：__reduce_and_partition_ipg_grads
         - __avg_scatter_grads - Reduce-Scatter 通信
         - partition_grads - 将分区梯度存储到目标缓冲区
  ↓
  2. 独立参数梯度聚合完成
  ↓
backward_epilogue (NVTX 标记)
  ↓
  - 如果启用 swap_optimizer：optimizer_swapper.post_backward()
  ↓
Backward 完成
```

### 详细步骤分析

#### Step 1: 梯度聚合准备 (__add_grad_to_ipg_bucket)

**位置：** `stage3.py:1406`

**操作流程：**

```python
# stage3.py:1406-1429
def __add_grad_to_ipg_bucket(self, param):
    bucket = self.ipg_buckets[self.get_param_comm_dtype(param)]
    
    # 将梯度复制到连续缓冲区
    if bucket.elements + param.grad.numel() <= self.reduce_bucket_size:
        new_grad_tensor = bucket.buffer.narrow(
            0, bucket.elements, param.grad.numel()
        ).view_as(param.grad)
        new_grad_tensor.copy_(param.grad, non_blocking=True)
        
        # 更新参数梯度指向
        param.grad.data = new_grad_tensor
    
    bucket.params.append(param)
    bucket.elements += param.grad.numel()
```

**Tensor 存储位置：**
- `param.grad`：GPU → 指向 IPG bucket 缓冲区
- `ipg_buckets[dtype].buffer`：GPU

**NVTX 标记：**
- `__add_grad_to_ipg_bucket` - 添加梯度到聚合 bucket

#### Step 2: 梯度 Reduce-Scatter (__avg_scatter_grads)

**位置：** `stage3.py:1639`

**操作流程：**

```python
# stage3.py:1639-1668
def __avg_scatter_grads(self, params_to_reduce, communication_data_type):
    full_grads_for_rank = [p.grad for p in params_to_reduce]
    
    # 数据类型转换
    if communication_data_type != self.dtype:
        full_grads_for_rank = [g.to(communication_data_type) for g in full_grads_for_rank]
    
    # 梯度预除
    if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
        full_grads_for_rank = [g.div(self.gradient_predivide_factor) for g in full_grads_for_rank]
    
    # Reduce-Scatter 通信
    grad_partitions_for_rank = reduce_scatter_coalesced(
        full_grads_for_rank, self.dp_process_group
    )
    
    # 梯度预缩放
    if self.postscale_gradients and self.gradient_predivide_factor != dist.get_world_size():
        grad_partitions_for_rank = [g.mul(self.gradient_predivide_factor) for g in grad_partitions_for_rank]
    
    return grad_partitions_for_rank
```

**Reduce-Scatter 实现：**
```python
# comm/coalesced_collectives.py
def reduce_scatter_coalesced(tensor_list, group):
    # 拼接张量
    coalesced_tensor = torch.cat(tensor_list)
    
    # 分配输出缓冲区
    output = torch.empty_like(coalesced_tensor)
    
    # Reduce-Scatter 操作
    dist.reduce_scatter_tensor(coalesced_tensor, output, group)
    
    # 拆分输出
    output_tensors = split_tensor(output, tensor_sizes)
    
    return output_tensors
```

**Tensor 存储位置：**
- 输入梯度：GPU
- 输出分区梯度：GPU
- Reduce-Scatter 缓冲区：GPU

**通信操作：**
- **GPU-GPU 集合通信**：
  - `dist.reduce_scatter_tensor()` - 所有 rank 的完整梯度 → 每个 rank 获得梯度分区

**NVTX 标记：**
- `__avg_scatter_grads` - 梯度平均和分区

#### Step 3: 梯度分区存储 (partition_grads)

**位置：** `stage3.py:1729`

**操作流程：**

```python
# stage3.py:1729-1791
def partition_grads(self, params_to_release, grad_partitions):
    for param, grad_partition in zip(params_to_release, grad_partitions):
        # 获取梯度分区缓冲区
        grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(
            0, 0, grad_partition.numel()
        )
        
        # 第一个 micro-batch：直接复制
        if self.micro_step_id == 0:
            grad_buffer.copy_(grad_partition, non_blocking=True)
            grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
        # 后续 micro-batch：累积
        else:
            if get_accelerator().on_accelerator(grad_buffer):
                grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
            else:  # CPU offload
                cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
                grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
        
        # 如果启用优化器卸载
        if self.offload_optimizer:
            if self.is_gradient_accumulation_boundary:
                # 计算梯度范数（用于梯度裁剪）
                self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)
            
            if self._swappable_optimizer_subgroup(i):
                # 准备卸载到 NVMe
                offload_fp32_gradients[i].append(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
            else:
                # 直接复制到 FP32 参数的 grad 属性
                fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(...)
                fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
        
        # 释放参数梯度
        param.grad = None
    
    # 如果启用 NVMe 卸载，将梯度卸载到 NVMe
    if self.offload_optimizer and self.swap_optimizer:
        for i in offload_fp32_gradients.keys():
            self.optimizer_swapper.swap_out_gradients(
                parameter=self.fp32_partitioned_groups_flat[i],
                gradient_offsets=offload_fp32_offsets[i],
                gradient_tensors=offload_fp32_gradients[i]
            )
```

**Tensor 存储位置：**
- 输入分区梯度 (`grad_partition`)：GPU
- 目标缓冲区（未卸载）：
  - `self.__param_id_to_grad_partition[param.ds_id]`：GPU
  - `self.fp32_partitioned_groups_flat[i].grad`：GPU
- 目标缓冲区（CPU 卸载）：
  - `self.grad_partitions_flat_buffer`：CPU (Pinned)
- 目标缓冲区（NVMe 卸载）：
  - NVMe 文件系统（通过 swap_out_gradients）

**通信操作：**
- **HtoD 通信**（CPU 卸载）：从 CPU 复制到 GPU（用于计算）
- **DtoH 通信**（NVMe 卸载）：从 GPU 复制到 NVMe（swap_out_gradients）

**NVTX 标记：**
- `partition_grads` - 梯度分区

#### Step 4: 梯度聚合完成 (independent_gradient_partition_epilogue)

**位置：** `stage3.py:1273`

**操作流程：**

```python
# stage3.py:1273-1300
def independent_gradient_partition_epilogue(self):
    # 处理所有 dtype 的 IPG bucket
    for comm_dtype in sort_dtypes(self.ipg_buckets.keys()):
        self.__reduce_and_partition_ipg_grads(comm_dtype)
    
    # 同步 Reduce-Scatter 流
    if not get_accelerator().resolves_data_dependency():
        self.reduce_and_partition_stream.synchronize()
    
    # 重置参数标志
    for param_id in self.params_already_reduced.keys():
        self.params_already_reduced[param_id] = False
    
    # 如果未启用优化器卸载，准备 averaged_gradients
    if not self.offload_optimizer:
        for i, sub_group in enumerate(self.fp16_groups):
            self.averaged_gradients[i] = [
                self.__param_id_to_grad_partition[param.ds_id]
                if param.requires_grad else torch.zeros_like(param.ds_tensor)
                for param in sub_group
            ]
```

**NVTX 标记：**
- `independent_gradient_partition_epilogue` - 独立参数梯度聚合完成

---

## Optimizer Step 阶段流程

### 流程概览

```
开始 Optimizer Step
  ↓
_pre_step
  ↓
_partition_all_parameters
  ↓
  - 释放所有参数分区（返回到 NOT_AVAILABLE 状态）
  ↓
_overflow_check_and_loss_scale_update
  ↓
  - 检查梯度是否溢出（inf/nan）
  - 调整 loss scale（如果使用动态 loss scaling）
  ↓
计算梯度范数
  ↓
遍历每个子组 (sub_group_id):
  ↓
  _prepare_sub_group (NVTX 标记)
  ↓
    1. 如果启用优化器卸载：
       _optimizer_states_and_gradient_swap_in
         - 如果参数在 NVMe：swap_in (Nvme→CPU→GPU)
         - 如果参数在 CPU：CPU→GPU
  ↓
    2. 如果未启用优化器卸载：
       _prepare_fp32_grad_for_sub_group
         - 将分区梯度从 GPU 复制到 FP32 参数的 grad
  ↓
  unscale_and_clip_grads
  ↓
    - 反缩放梯度（除以 loss_scale）
    - 梯度裁剪（如果配置）
  ↓
  _optimizer_step (NVTX 标记)
  ↓
    - 执行优化器 step（如 Adam）
    - FP32 参数 ← FP32 参数 - lr * gradient
  ↓
  _reassign_or_swap_out_partitioned_parameters (NVTX 标记)
  ↓
    1. 如果参数在 GPU：
       - 将 FP32 参数复制回 FP16 参数分区
       - fp16_partitioned_groups_flat[i] ← fp32_partitioned_groups_flat[i]
  ↓
    2. 如果参数在 NVMe：
       _partitioned_params_swap_out
         - 将 FP32 参数写入 NVMe
  ↓
  _release_sub_group (NVTX 标记)
  ↓
    1. 释放 FP32 梯度
    2. 如果启用优化器卸载：
       _optimizer_states_and_gradient_swap_out
         - 将 FP32 参数、梯度、momentum、variance 卸载到 NVMe
  ↓
_post_step
  ↓
  - 持久化参数的 all_gather（如果有）
  ↓
Optimizer Step 完成
```

### 详细步骤分析

#### Step 1: 准备子组 (_prepare_sub_group)

**位置：** `stage3.py:2260`

**操作流程：**

```python
# stage3.py:2260-2266
def _prepare_sub_group(self, sub_group_id, timer_names):
    # 如果启用优化器卸载
    if self._swappable_optimizer_subgroup(sub_group_id):
        self._optimizer_states_and_gradient_swap_in(sub_group_id, timer_names)
    # 如果未启用优化器卸载
    elif not self.offload_optimizer:
        self._prepare_fp32_grad_for_sub_group(sub_group_id)
```

**子步骤 1: 优化器状态和梯度 swap-in**

**位置：** `stage3.py:2268`

```python
# stage3.py:2268-2285
def _optimizer_states_and_gradient_swap_in(self, sub_group_id, timer_names):
    param_length = self.fp16_partitioned_groups_flat_numel[sub_group_id]
    fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
    
    # 从 NVMe 交换到 GPU
    self.optimizer_swapper.swap_in_optimizer_state(
        parameter=fp32_param,
        async_parameter=self.next_swappable_fp32_partitioned_groups[sub_group_id]
    )
```

**Swap-In 实现：**
```python
# partitioned_optimizer_swapper.py:73-99
def swap_in_optimizer_state(self, parameter, async_parameter):
    swap_info = self._get_param_swap_info(parameter)
    
    # 分配 pinned 缓冲区
    pinned_buffers = self.swap_buffer_manager.allocate(...)
    swap_info.swap_buffers = pinned_buffers.copy()
    
    # 参数 swap-in
    self._swap_in_parameter(aio_handle=self.aio_handle, parameter=parameter, ...)
    
    # 梯度 swap-in（如果有）
    if swap_info.has_gradients():
        self._swap_in_gradients(aio_handle=self.aio_handle, parameter=parameter, ...)
```

**Swap-In 参数：**
```python
# partitioned_optimizer_swapper.py:165-196
def _swap_in_parameter(self, aio_handle, parameter, dest_buffers):
    swap_info = self._get_param_swap_info(parameter)
    
    # 分配缓冲区
    swap_buffers = get_sized_buffers(dest_buffers, swap_lengths)
    compute_buffers = get_sized_buffers(dest_buffers, compute_lengths)
    
    # 从 NVMe 读取
    swap_in_tensors(aio_handle, swap_buffers, swap_info.get_swap_paths())
    
    # 等待 I/O 完成
    aio_handle.wait()
    
    # 设置交换缓冲区
    swap_info.set_swap_buffers(dest_buffers, ...)
```

**Tensor 存储位置：**
- **优化器参数**：
  - 输入：NVMe / CPU
  - 输出：GPU
- **优化器梯度**：
  - 输入：NVMe / CPU
  - 输出：GPU

**通信操作：**
- **DtoH 通信**（NVMe 卸载）：从 NVMe 读取到 CPU
- **HtoD 通信**：从 CPU 复制到 GPU

**NVTX 标记：**
- `_optimizer_states_and_gradient_swap_in` - 优化器状态和梯度 swap-in
- `swap_in_optimizer_state` - 优化器参数 swap-in
- `swap_in_gradient` - 梯度 swap-in

**子步骤 2: 准备 FP32 梯度（未卸载）**

```python
# stage3.py (隐含在 partition_grads 中)
def _prepare_fp32_grad_for_sub_group(self, sub_group_id):
    for param in self.fp16_groups[sub_group_id]:
        if param.requires_grad:
            grad_partition = self.averaged_gradients[sub_group_id][i]
            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            
            # 计算偏移和长度
            offset, length = self.grad_position[param.ds_id]
            
            # 复制梯度分区到 FP32 参数的 grad 属性
            fp32_grad = fp32_param.grad.narrow(0, offset, length)
            fp32_grad.copy_(grad_partition)
```

**Tensor 存储位置：**
- 输入：`averaged_gradients` (GPU)
- 输出：`fp32_partitioned_groups_flat[i].grad` (GPU)

#### Step 2: 反缩放和梯度裁剪 (unscale_and_clip_grads)

**位置：** `stage3.py:2536

**操作流程：**

```python
# stage3.py:2536-2545
def unscale_and_clip_grads(self, sub_group_id, total_norm):
    # 计算组合缩放因子
    combined_scale = self.loss_scale
    if self.clip_grad > 0.:
        clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
        clip = torch.clamp(clip, min=1.0)
        combined_scale = clip * self.loss_scale
    
    # 反缩放梯度
    self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)
```

**Tensor 存储位置：**
- 输入/输出：`fp32_partitioned_groups_flat[i].grad` (GPU)

**NVTX 标记：**
- `unscale_and_clip_grads` - 反缩放和梯度裁剪

#### Step 3: 执行优化器 step (_optimizer_step)

**位置：** `stage3.py:1103

**操作流程：**

```python
# stage3.py:1103-1126
def _optimizer_step(self, sub_group_id):
    param_group_id = self.sub_group_to_group_id[sub_group_id]
    fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
    
    # 设置优化器参数
    if self.offload_optimizer:
        cur_device = self.subgroup_to_device[sub_group_id]
        if cur_device == 'cpu':
            self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            optimizer.step()  # CPU Adam
            self.optimizer.param_groups[param_group_id]['params'] = []
        else:
            self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            backup_optimizer.step()  # GPU 备份优化器
            self.backup_optimizer.param_groups[param_group_id]['params'] = []
    else:
        self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
        optimizer.step()  # GPU 优化器
        self.optimizer.param_groups[param_group_id]['params'] = []
```

**优化器 step 示例（Adam）：**
```
FP32 参数更新：
- moment ← β1 * moment + (1 - β1) * gradient
- variance ← β2 * variance + (1 - β2) * gradient²
- update ← moment / (√variance + ε)
- FP32 参数 ← FP32 参数 - lr * update
```

**Tensor 存储位置：**
- **FP32 参数**：GPU / CPU
- **梯度**：GPU / CPU
- **Moment**：GPU / CPU
- **Variance**：GPU / CPU

**NVTX 标记：**
- `step_with_gradscaler` - 使用梯度缩放器的优化器 step
- `optimizer.step` - 优化器 step（具体优化器实现）

#### Step 4: 重新分配或交换出分区参数 (_reassign_or_swap_out_partitioned_parameters)

**位置：** `stage3.py:2424

**操作流程：**

```python
# stage3.py:2424-2435
def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
    # 如果参数在 GPU
    if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
        # 将 FP32 参数复制到 FP16 参数分区
        self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
            self.fp32_partitioned_groups_flat[sub_group_id].data
        )
        
        # 恢复 FP16 参数形状
        self._unflatten_partitioned_parameters(sub_group_id)
    # 如果参数在 NVMe
    else:
        self._partitioned_params_swap_out(sub_group_id)
```

**子步骤 1: NVMe swap-out**

**位置：** `stage3.py:1135

```python
# stage3.py:1135-1154
def _partitioned_params_swap_out(self, i):
    fp32_param = self.fp32_partitioned_groups_flat[i]
    
    swap_fp16_params = []
    swap_fp32_params = []
    
    for param, partitioned_param in zip(self.fp16_groups[i], self.fp16_partitioned_groups[i]):
        src = fp32_param.narrow(0, offset, partitioned_param.ds_numel)
        
        # 如果参数在 GPU
        if partitioned_param.status == PartitionedParamStatus.AVAILABLE:
            partitioned_param.data.copy_(src.data)
        # 如果参数在 NVMe
        else:
            swap_fp32_params.append(src)
            swap_fp16_params.append(param)
    
    # 将 FP32 参数写入 NVMe
    if len(swap_fp16_params):
        swap_fp16_params[0].nvme_swapper.swap_out_partitioned_params(
            dst_fp16_params=swap_fp16_params,
            src_fp32_params=swap_fp32_params
        )
```

**Tensor 存储位置：**
- 输入：`fp32_partitioned_groups_flat[i]` (GPU)
- 输出（GPU）：`fp16_partitioned_groups_flat[i]` (GPU)
- 输出（NVMe）：NVMe 文件系统

**通信操作：**
- **DtoH 通信**（NVMe 卸载）：从 GPU 复制到 NVMe

**NVTX 标记：**
- `_reassign_or_swap_out_partitioned_parameters` - 重新分配或交换出分区参数
- `_partitioned_params_swap_out` - NVMe 参数交换出

#### Step 5: 释放子组 (_release_sub_group)

**位置：** `stage3.py:2288

**操作流程：**

```python
# stage3.py:2288-2296
def _release_sub_group(self, sub_group_id, timer_names):
    # 释放 FP32 梯度
    if not self.offload_optimizer:
        self.fp32_partitioned_groups_flat[sub_group_id].grad = None
    
    # 如果启用优化器卸载
    if self._swappable_optimizer_subgroup(sub_group_id):
        self._optimizer_states_and_gradient_swap_out(sub_group_id, timer_names)
```

**优化器状态 swap-out：**

**位置：** `stage3.py:2318`

```python
# stage3.py:2318-2335
def _optimizer_states_and_gradient_swap_out(self, sub_group_id, timer_names):
    fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
    
    # 将优化器状态交换到 NVMe
    self.optimizer_swapper.swap_out_optimizer_state(
        parameter=fp32_param,
        async_swap=self.next_swappable_fp32_partitioned_groups[sub_group_id] is not None
    )
    
    # 释放 FP32 梯度
    self.fp32_partitioned_groups_flat[sub_group_id].grad = None
```

**Swap-Out 实现：**
```python
# partitioned_optimizer_swapper.py:141-154
def swap_out_optimizer_state(self, parameter, async_swap):
    swap_info = self._get_param_swap_info(parameter)
    
    # 计算交换字节数
    swap_bytes = sum([...])
    
    # 交换出
    self._swap_out_optimizer_state(swap_info)
    
    # 释放交换缓冲区
    self.release_swap_buffers(parameter)
```

**Tensor 释放：**
- `fp32_partitioned_groups_flat[i].grad`：设置为 None
- 优化器状态缓冲区：释放或写入 NVMe

**通信操作：**
- **DtoH 通信**（NVMe 卸载）：从 GPU 复制到 NVMe

**NVTX 标记：**
- `_release_sub_group` - 释放优化器子组
- `_optimizer_states_and_gradient_swap_out` - 优化器状态和梯度 swap-out
- `swap_out_optimizer_state` - 优化器参数 swap-out

---

## 通信操作汇总

### GPU-GPU 集合通信

| 操作 | 描述 | 位置 | 阶段 |
|------|------|------|------|
| `dist.all_gather()` | 所有 rank 的参数分区 → 每个 rank 获得完整参数 | partition_parameters.py:1298 | Forward |
| `dist.all_reduce()` | 所有 rank 的参数分区求和 → 每个 rank 获得完整参数 | partition_parameters.py:1365 | Forward |
| `dist.reduce_scatter_tensor()` | 所有 rank 的完整梯度 → 每个 rank 获得梯度分区 | comm/coalesced_collectives.py | Backward |
| `dist.all_reduce()` | 溢出检测（所有 rank 的溢出标志取最大值）| stage3.py:2584 | Optimizer Step |

### GPU-CPU 通信 (DtoH / HtoD)

| 操作 | 方向 | 描述 | 位置 | 阶段 |
|------|------|------|------|------|
| `.to("cuda")` | HtoD | 从 CPU 复到 GPU | partition_parameters.py:1298 | Forward |
| `.to("cpu")` | DtoH | 从 GPU 复到 CPU | stage3.py:1766 | Backward |
| `.copy_(...)` | HtoD | 从 CPU 复到 GPU | stage3.py:1768 | Backward |
| `torch.cat()` (Pinned) | HtoD | 从 Pinned CPU 复到 GPU | stage3.py:1810 | Backward |
| `swap_in()` | DtoH→HtoD | NVMe → CPU → GPU | partitioned_optimizer_swapper.py:73 | Optimizer Step |
| `swap_out()` | HtoD→DtoH | GPU → CPU → NVMe | partitioned_optimizer_swapper.py:141 | Optimizer Step |

### NVMe I/O 通信

| 操作 | 方向 | 描述 | 位置 | 阶段 |
|------|------|------|------|------|
| `swap_in()` | NVMe → CPU | 从 NVMe 读取参数 | partitioned_param_swapper.py | Forward |
| `swap_out()` | CPU → NVMe | 将参数写入 NVMe | partitioned_param_swapper.py | Backward |
| `swap_in_optimizer_state()` | NVMe → CPU → GPU | 从 NVMe 读取优化器状态 | partitioned_optimizer_swapper.py:73 | Optimizer Step |
| `swap_out_optimizer_state()` | GPU → CPU → NVMe | 将优化器状态写入 NVMe | partitioned_optimizer_swapper.py:141 | Optimizer Step |
| `swap_in_gradients()` | NVMe → CPU → GPU | 从 NVMe 读取梯度 | partitioned_optimizer_swapper.py:198 | Optimizer Step |
| `swap_out_gradients()` | GPU → CPU → NVMe | 将梯度写入 NVMe | partitioned_optimizer_swapper.py:159 | Optimizer Step |

---

## Micro-Batch 处理差异 (GAS=4)

假设梯度累积步数 (GAS) = 4，即 4 个 micro-batch 累积梯度后执行一次 optimizer step。

### 第一个 Micro-Batch (Micro-Step 0)

**特点：**
- 梯度缓冲区为空，直接存储梯度分区
- 不需要从 NVMe/CPU 加载历史梯度

**详细流程：**

1. **Forward 阶段**
   - 与普通 forward 相同，无特殊处理

2. **Backward 阶段**
   - 梯度计算完成后，触发 gradient hook
   - `reduce_partition_and_remove_grads()` → `reduce_independent_p_g_buckets_and_remove_grads()`
   - `__add_grad_to_ipg_bucket()` - 将梯度添加到 IPG bucket
   - `__reduce_and_partitioned_grads()` - Reduce-Scatter 和分区
   - `partition_grads()` - 存储梯度分区

   **关键代码：**
   ```python
   # stage3.py:1744-1759
   def partition_grads(self, params_to_release, grad_partitions):
       for param, grad_partition in zip(params_to_release, grad_partitions):
           grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(...)
           
           # 第一个 micro-batch：直接复制
           if self.micro_step_id == 0:  # ✅ 条件成立
               grad_buffer.copy_(grad_partition, non_blocking=True)
               grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
           else:  # ❌ 不执行
               grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
   ```

   **Tensor 处理：**
   - 输入：`grad_partition` (GPU)
   - 输出：`grad_buffer` (GPU / CPU Pinned)
   - 操作：直接复制（`copy_`）

3. **Optimizer Step 阶段**
   - 不执行（`is_gradient_accumulation_boundary` = False）

### 第二/三/四个 Micro-Batch (Micro-Step 1/2/3)

**特点：**
- 梯度缓冲区已有数据，需要累积梯度
- 如果启用 NVMe 卸载，可能需要从 NVMe 加载梯度分区

**详细流程：**

1. **Forward 阶段**
   - 与普通 forward 相同，无特殊处理

2. **Backward 阶段**
   - 梯度计算完成后，触发 gradient hook
   - `reduce_partition_and_remove_grads()` → `reduce_independent_p_g_buckets_and_remove_grads()`
   - `__add_grad_to_ipg_bucket()` - 将梯度添加到 IPG bucket
   - `__reduce_and_partition_ipg_grads()` - Reduce-Scatter 和分区
   - `partition_grads()` - 累积梯度分区

   **关键代码：**
   ```python
   # stage3.py:1744-1759
   def partition_grads(self, params_to_release, grad_partitions):
       for param, grad_partition in zip(params_to_release, grad_partitions):
           grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(...)
           
           # 第一个 micro-batch：直接复制
           if self.micro_step_id == 0:  # ❌ 条件不成立
               grad_buffer.copy_(grad_partition, non_blocking=True)
           else:  # ✅ 执行累积
               # 如果梯度缓冲区在 GPU
               if get_accelerator().on_accelerator(grad_buffer):
                   grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
               # 如果梯度缓冲区在 CPU (CPU 卸载)
               else:
                   # 1. 先复制到 GPU
                   cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                   # 2. 在 GPU 上累积
                   cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
                   # 3. 复制回 CPU
                   grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
   ```

   **Tensor 处理：**
   - 输入：`grad_partition` (GPU)
   - 输出：`grad_buffer` (GPU / CPU Pinned)
   - 操作：
     - 未卸载：GPU 上的 `add_` 操作
     - CPU 卸载：GPU→D→GPU 累积→DtoH→CPU

   **通信操作：**
   - **CPU 卸载**：
     - HtoD：`grad_buffer.to(grad_partition.device)` - CPU → GPU
     - DtoH：`grad_buffer.copy_(cuda_grad_buffer)` - GPU → CPU

3. **Optimizer Step 阶段**
   - 不执行（`is_gradient_accumulation_boundary` = False）

### 最后一个 Micro-Batch (Micro-Step 4，GAS 边界)

**特点：**
- 梯度累积完成，执行 optimizer step
- 需要从 NVMe/CPU 加载优化器状态（FP32 参数、momentum、variance）
- Optimizer step 完成后，将更新后的参数写回 NVMe/CPU

**详细流程：**

1. **Forward 阶段**
   - 与普通 forward 相同，无特殊处理

2. **Backward 阶段**
   - 梯度计算完成后，触发 gradient hook
   - `reduce_partition_and_remove_grads()` → `reduce_independent_p_g_buckets_and_remove_grads()`
   - `__add_grad_to_ipg_bucket()` - 将梯度添加到 IPG bucket
   - `__reduce_and_partition_ipg_grads()` - Reduce-Scatter 和分区
   - `partition_grads()` - 累积梯度分区

   **关键代码：**
   ```python
   # stage3.py:1744-1791
   def partition_grads(self, params_to_release, grad_partitions):
       for param, grad_partition in zip(params_to_release, grad_partitions):
           grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(...)
           
           # 累积梯度（同第二个 micro-batch）
           if self.micro_step_id == 0:  # ❌ 条件不成立
               grad_buffer.copy_(grad_partition, non_blocking=True)
           else:  # ✅ 执行累积
               if get_accelerator().on_accelerator(grad_buffer):
                   grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
               else:
                   cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                   cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype))
                   grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
           
           # ✅ 关键差异：GAS 边界，计算梯度范数并准备卸载
           if self.offload_optimizer:
               if self.is_gradient_accumulation_boundary:  # ✅ 条件成立
                   # 计算梯度范数（用于梯度裁剪）
                   self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)
                   
                   if self._swappable_optimizer_subgroup(i):
                       # 准备卸载到 NVMe
                       offload_fp32_gradients[i].append(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
                   else:
                       # 直接复制到 FP32 参数的 grad 属性
                       fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(...)
                       fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
       
       # ✅ 关键差异：如果启用 NVMe 卸载，将梯度卸载到 NVMe
       if self.offload_optimizer and self.swap_optimizer:
           for i in offload_fp32_gradients.keys():
               self.optimizer_swapper.swap_out_gradients(...)
   ```

   **Tensor 处理：**
   - 输入：`grad_partition` (GPU)
   - 输出（未卸载）：`fp32` 梯度缓冲区
   - 输出（NVMe 卸载）：NVMe 文件系统
   - 操作：
     - 未卸载：累积后复制到 FP32 梯度缓冲区
     - CPU 卸载：累积后复制到 FP32 梯度缓冲区（GPU），计算梯度范数，保留在 GPU
     - NVMe 卸载：累积后复制到 FP32 梯度缓冲区（GPU），计算梯度范数，swap-out 到 NVMe

   **通信操作：**
   - **NVMe 卸载**：
     - DtoH：GPU → CPU → NVMe（swap_out_gradients）

3. **Optimizer Step 阶段**
   - ✅ 执行（`is_gradient_accumulation_boundary` = True）

   **详细流程：**

   ```python
   # stage3.py:2444-2488
   def step(self, closure=None):
       self._pre_step()
       self._partition_all_parameters()
       
       # 检查溢出和更新 loss scale
       if self._overflow_check_and_loss_scale_update():
           return
       
       # 计算梯度范数
       norm_groups = self._get_norm_groups()
       scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))
       
       # 遍历每个子组
       for sub_group_id, group in enumerate(self.fp16_groups):
           # ✅ 关键步骤：准备子组
           self._prepare_sub_group(sub_group_id, timer_names)
           
           # 反缩放和裁剪梯度
           self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)
           
           # 执行优化器 step
           self._optimizer_step(sub_group_id)
           
           # ✅ 关键步骤：重新分配或交换出分区参数
           self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
           
           # ✅ 关键步骤：释放子组
           self._release_sub_group(sub_group_id, timer_names)
       
       self._post_step(timer_names)
   ```

   **子步骤 1: _prepare_sub_group**
   ```python
   # stage3.py:2260-2266
   def _prepare_sub_group(self, sub_group_id, timer_names):
       # ✅ 如果启用优化器卸载
       if self._swappable_optimizer_subgroup(sub_group_id):
           self._optimizer_states_and_gradient_swap_in(sub_group_id, timer_names)
           # - NVMe → CPU → GPU（swap_in_optimizer_state）
       elif not self.offload_optimizer:
           self._prepare_fp32_grad_for_sub_group(sub_group_id)
           # - 已在 partition_grads 中准备好
   ```

   **Tensor 处理：**
   - 输入（NVMe）：优化器状态（FP32 参数、momentum、variance）在 NVMe
   - 输出（NVMe）：优化器状态在 GPU

   **通信操作：**
   - **NVMe 卸载**：
     - DtoH→HtoD：NVMe → CPU → GPU（swap_in_optimizer_state）

   **子步骤 2: _optimizer_step**
   ```python
   # stage3.py:1103-1126
   def _optimizer_step(self, sub_group_id):
       fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
       
       # 设置优化器参数
       if self.offload_optimizer:
           cur_device = self.subgroup_to_device[sub_group_id]
           if cur_device == 'cpu':
               self.optimizer.param_groups['params'] = [fp32_param]
               optimizer.step()  # ✅ CPU Adam
           else:
               self.backup_optimizer.param_groups['params'] = [fp32_param]
               backup_optimizer.step()  # ✅ GPU 备份优化器
       else:
           self.optimizer.param_groups['params'] = [fp32_param]
           optimizer.step()  # ✅ GPU 优化器
   ```

   **Tensor 处理：**
   - 优化器 step 在 GPU 或 CPU 上执行

   **子步骤 3: _reassign_or_swap_out_partitioned_parameters**
   ```python
   # stage3.py:2424-2435
   def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
       # 如果参数在 GPU
       if self.fp16_partitioned_flat[sub_group_id] is not None:
           # ✅ 将 FP32 参数复制回 FP16 参数分区
           self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
               self.fp32_partitioned_groups_flat[sub_group_id].data
           )
           
           # 恢复 FP16 参数形状
           self._unflatten_partitioned_parameters(sub_group_id)
       # 如果参数在 NVMe
       else:
           # ✅ 将 FP32 参数写入 NVMe
           self._partitioned_params_swap_out(sub_group_id)
   ```

   **Tensor 处理：**
   - 输入：`fp32_partitioned_groups_flat[i]` (GPU)
   - 输出（GPU）：`fp16_partitioned_groups_flat[i]` (GPU)
   - 输出（NVMe）：NVMe 文件系统

   **通信操作：**
   - **NVMe 卸载**：
     - DtoH：GPU → CPU → NVMe（swap_out_partitioned_params）

   **子步骤 4: _release_sub_group**
   ```python
   # stage3.py:2288-2296
   def _release_sub_group(self, sub_group_id, timer_names):
       # 释放 FP32 梯度
       if not self.offload_optimizer:
           self.fp32_partitioned_groups_flat[sub_group_id].grad = None
       
       # ✅ 如果启用优化器卸载
       if self._swappable_optimizer_subgroup(sub_group_id):
           self._optimizer_states_and_gradient_swap_out(sub_group_id, timer_names)
           # - GPU → CPU → NVMe（swap_out_optimizer_state）
   ```

   **Tensor 处理：**
   - 输入（NVMe）：优化器状态（FP32 参数、momentum、variance）在 GPU
   - 输出（NVMe）：优化器状态在 NVMe

   **通信操作：**
   - **NVMe 卸载**：
     - HtoD→DtoH：GPU → CPU → NVMe（swap_out_optimizer_state）

### Micro-Batch 处理差异总结

| 阶段 | 第一个 Micro-Batch | 第二/三/四个 Micro-Batch | 最后一个 Micro-Batch (GAS 边界) |
|------|-------------------|----------------------|------------------------|
| **Forward** | 无差异 | 无差异 | 无差异 |
| **Backward - 梯度存储** | 直接复制（`copy_`）| 累积（`add_`）| 累积（`add_`）+ 计算梯度范数 |
| **Backward - 梯度卸载** | 不执行 | 不执行 | ✅ 执行（swap_out_gradients）|
| **Optimizer Step** | 不执行 | 不执行 | ✅ 执行完整流程 |
| **CPU 卸载通信** | 无 | HtoD→GPU 累积→DtoH | HtoD→GPU 累积→DtoH |
| **NVMe 卸载通信** | 无 | 无 | DtoH→HtoD（swap_in）+ HtoD→DtoH（swap_out）|

---

## 附录

### A. 关键文件路径

| 文件 | 描述 |
|------|------|
| `deepspeed/runtime/zero/stage3.py` | ZeRO Stage 3 主实现 |
| `deepspeed/runtime/zero/parameter_offload.py` | 参数卸载和模块钩子 |
| `deepspeed/runtime/zero/partition_parameters.py` | 参数分区实现 |
| `deepspeed/runtime/zero/partitioned_param_coordinator.py` | 参数协调器（获取、释放、预取）|
| `deepspeed/runtime/zero/offload_config.py` | 卸载配置定义 |
| `deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py` | 优化器状态交换（NVMe）|
| `deepspeed/runtime/swap_tensor/partitioned_param_swapper.py` | 参数交换（NVMe）|
| `deepspeed/runtime/comm/coalesced_collectives.py` | 聚合通信（Reduce-Scatter）|

### B. 关键函数索引

| 函数名 | 位置 | 描述 |
|--------|------|------|
| `fetch_sub_module` | parameter_offload.py:295 | 获取子模块参数 |
| `release_sub_module` | parameter_offload.py:470 | 释放子模块参数 |
| `all_gather` | partition_parameters.py:1227 | All-Gather 参数 |
| `partition` | partition_parameters.py:1494 | 分区参数 |
| `reduce_partition_and_remove_grads` | stage3.py:1327 | 梯度聚合钩子 |
| `__add_grad_to_ipg_bucket` | stage3.py:1406 | 添加梯度到聚合 bucket |
| `__reduce_and_partition_ipg_grads` | stage3.py:1431 | 梯度 Reduce-Scatter 和分区 |
| `__avg_scatter_grads` | stage3.py:1639 | 梯度平均和分区 |
| `partition_grads` | stage3.py:1729 | 存储梯度分区 |
| `step` | stage3.py:2444 | 优化器 step |
| `_prepare_sub_group` | stage3.py:2260 | 准备优化器子组 |
| `_optimizer_states_and_gradient_swap_in` | stage3.py:2268 | Swap-in 优化器状态 |
| `_optimizer_step` | stage3.py:1103 | 执行优化器 step |
| `_reassign_or_swap_out_partitioned_parameters` | stage3.py:2424 | 重新分配或 swap-out 参数 |
| `_release_sub_group` | stage3.py:2288 | 释放优化器子组 |
| `_optimizer_states_and_gradient_swap_out` | stage3.py:2318 | Swap-out 优化器状态 |

### C. NVTX 标记汇总

| 标记 | 位置 | 阶段 |
|------|------|------|
| `pre_sub_module_forward_function` | parameter_offload.py:465 | Forward |
| `post_sub_module_forward_function` | parameter_offload.py:484 | Forward |
| `pre_sub_module_backward_function` | parameter_offload.py:394 | Backward |
| `post_sub_module_backward_function` | parameter_offload.py:430 | Backward |
| `reduce_partition_and_remove_grads` | stage3.py:1327 | Backward |
| `__add_grad_to_ipg_bucket` | stage3.py:1406 | Backward |
| `__reduce_and_partition_ipg_grads` | stage3.py:1431 | Backward |
| `partition_grads` | stage3.py:1729 | Backward |
| `independent_gradient_partition_epilogue` | stage3.py:1273 | Backward |
| `step` | stage3.py:2444 | Optimizer Step |
| `_prepare_sub_group` | stage3.py:2260 | Optimizer Step |
| `_optimizer_states_and_gradient_swap_in` | stage3.py:2268 | Optimizer Step |
| `unscale_and_clip_grads` | stage3.py:2536 | Optimizer Step |
| `_optimizer_step` | stage3.py:1103 | Optimizer Step |
| `_reassign_or_swap_out_partitioned_parameters` | stage3.py:2424 | Optimizer Step |
| `_partitioned_params_swap_out` | stage3.py:1135 | Optimizer Step |
| `_release_sub_group` | stage3.py:2288 | Optimizer Step |
| `_optimizer_states_and_gradient_swap_out` | stage3.py:2318 | Optimizer Step |
