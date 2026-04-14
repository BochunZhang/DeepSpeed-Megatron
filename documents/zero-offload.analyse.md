# DeepSpeed Zero-Offload 分析文档

## 1. 脚本配置分析

根据 `finetune_qwen3-14b_4gpu.sh` 脚本中的 `zerooffload` 模式配置：

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin": true
        }
    },
    "wall_clock_breakdown": true
}
```

**关键配置参数：**
- **stage**: 3 - 完全参数分区
- **offload_optimizer.device**: "cpu" - 优化器状态卸载到 CPU
- **offload_optimizer.pin**: true - 使用 pinned memory 加速 CPU-GPU 传输
- **overlap_comm**: false - 不重叠通信和计算
- **reduce_bucket_size**: 400MB - 梯度归约 bucket 大小
- **sub_group_size**: 4e8 - 参数子组大小
- **gradient_accumulation_steps**: 2 (根据 gbs=8, mbs=1, 4gpu 计算: 8/(1*4)=2)

---

## 2. Forward 阶段处理流程

### 2.1 整体流程

Forward 阶段按模块层级逐步执行，每个模块包括以下子阶段：

```
pre_sub_module_forward (前置处理)
  └─> fetch_sub_module (获取参数)
      └─> __all_gather_params (all-gather 通信)
  └─> 模块前向计算 (实际神经网络计算)
post_sub_module_forward (后置处理)
  └─> release_sub_module (释放参数)
```

### 2.2 详细步骤

#### 步骤 1: pre_sub_module_forward_function

**代码位置:** `deepspeed/runtime/zero/parameter_offload.py:464`
**NVTX 标记:** `pre_sub_module_forward_function`

- 将当前模块推入 `FWD_MODULE_STACK`
- 调用 `param_coordinator.trace_prologue(sub_module)` 记录模块访问模式
- 调用 `param_coordinator.fetch_sub_module(sub_module, forward=True)` 获取参数

#### 步骤 2: _fetch_sub_module_impl

**代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:292`
**NVTX 标记:** `_fetch_sub_module_impl`

**处理流程：**

1. **参数获取** (`_fetch_sub_module_impl`):
   - 识别当前模块需要的参数集合 `params_to_fetch`
   - 对于每个状态为 `NOT_AVAILABLE` 的参数，触发 all-gather

2. **All-Gather 通信** (`__all_gather_params_`):
   **代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:508`
   **NVTX 标记:** `__all_gather_params_`

   - 使用 `self.__allgather_stream` 独立 CUDA stream
   - 调用 `param.all_gather_coalesced(params)` 执行 all-gather
   - 将参数状态从 `NOT_AVAILABLE` 转为 `INFLIGHT`
   - 存储 `AllGatherCoalescedHandle` 到 `inflight_param_registry`

3. **参数预取** (prefetch):
   - 如果已完成模块 trace，根据 trace 预取即将使用的参数
   - 预取受 `prefetch_bucket_sz` 限制 (默认 50MB)
   - 受 `max_available_parameters` 限制避免内存溢出

4. **等待参数就绪**:
   - 对于 `params_to_fetch` 中的每个参数：
     - 如果参数在 `inflight_param_registry` 中，调用 `handle.wait()`
     - 等待完成后，参数状态变为 `AVAILABLE`

**通信详情：**
- **GPU 间通信:** All-gather 操作，将参数分区从所有 GPU 收集到当前 GPU
- **通信参数:** 所有 rank 参与，每个 rank 接收完整的参数（其拥有的分区 + 其他 rank 的分区）
- **NVTX 标记:** `FORWARD_FETCH_SUBMIT`, `FORWARD_FETCH_WAIT`, `FORWARD_PREFETCH_SUBMIT`, `FORWARD_ALL_GATHER`

#### 步骤 3: 模块前向计算

**NVTX 标记:** 无 (PyTorch 自身计算)

---

### 2.3 详细步骤续

#### 步骤 1 续: pre_sub_module_forward_function

**代码位置:** `deepspeed/runtime/zero/parameter_offload.py:464`
**NVTX 标记:** 无 (已说明)

- 将当前模块推入 `FWD_MODULE_STACK`
- 调用 `param_coordinator.trace_prologue(sub_module)` 记录模块访问模式
- 调用 `param_coordinator.fetch_sub_module(sub_module, forward=True)` 获取参数

#### 步骤 2 续: _fetch_sub_module_impl

**代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:292`
**NVTX 标记:** `_fetch_sub_module_impl`

**处理流程：**

1. **参数获取** (`_fetch_sub_module_impl`):
   - 识别当前模块需要的参数集合 `params_to_fetch`
   - 对于每个状态为 `NOT_AVAILABLE` 的参数，触发 all-gather

2. **All-Gather 通信** (`__all_gather_params_`):
   **代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:508`
   **NVTX 标记:** `__all_gather_params_`

   - 使用 `self.__allgather_stream` 独立 CUDA stream
   - 调用 `param.all_gather_coalesced(params)` 执行 all-gather
   - 将参数状态从 `NOT_AVAILABLE` 转为 `INFLIGHT`
   - 存储 `AllGatherCoalescedHandle` 到 `inflight_param_registry`

- PyTorch 执行实际的前向计算
- 参数现在以 `AVAILABLE` 状态存在，可直接访问
- 计算产生的中间激活值和输出

#### 步骤 4: post_sub_module_forward

**代码位置:** `deepspeed/runtime/zero/parameter_offload.py:484`
**NVTX 标记:** 无（装饰器内部的函数）

- 弹出 `FWD_MODULE_STACK` 中的当前模块
- 调用 `param_coordinator.release_sub_module(sub_module, forward=True)` 释放参数

#### 步骤 5: _release_param

**代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:468`
**NVTX 标记:** `_release_param`

**释放条件判断：**
- 如果已完成 trace: 使用 `__params_to_release()` 决定释放哪些参数
- 否则：释放当前模块的所有非持久参数

**释放逻辑** (`__release_param`):
**代码位置:** `deepspeed/runtime/zero/partitioned_param_coordinator.py:562`

- 检查参数状态是否为 `AVAILABLE` 且无活跃子模块
- 调用 `param.partition(free_data=True)` 释放底层存储
- 减少 `__n_available_params` 计数

---

## 3. Backward 阶段处理流程

### 3.1 整体流程

```
pre_sub_module_backward (前置处理)
  └─> fetch_sub_module (获取参数)
      └─> __all_gather_params (all-gather 通信)
  └─> 模块反向计算 (梯度计算)
post_sub_module_backward (后置处理)
  └─> release_sub_module (释放参数)
梯度归约和分区 (在 backward 完成后触发)
  └─> reduce_independent_p_g_buckets
      └─> __reduce_and_partition_ipg_grads
          └─> __avg_scatter_contiguous_grads 或 __avg_scatter_grads (reduce-scatter/all2all)
  └─> partition_grads (分区梯度并卸载到 CPU)
```

### 3.2 详细步骤

#### 步骤 1: pre_sub_module_backward_function

**代码位置:** `deepspeed/runtime/zero/parameter_offload.py:502`
**NVTX 标记:** `pre_sub_module_backward_function`

- 调用 `param_coordinator.trace_prologue(sub_module)`
- 调用 `param_coordinator.fetch_sub_module(sub_module, forward=False)`
- 注意：与 forward 相同，但 `forward=False` 标记

#### 步骤 2: 模块反向计算

- PyTorch 自动求导执行反向传播
- 生成参数梯度 (`param.grad`)
- 每个参数注册的 `reduce_partition_and_remove_grads` hook 将被触发

#### 步骤 3: 梯度归约 Hook

**代码位置:** `deepspeed/runtime/zero/stage3.py:1326`
**NVTX 标记:** 无（装饰器内部）

**Hook 函数：** `reduce_partition_and_remove_grads`

1. 调用 `self.reduce_ready_partitions_and_remove_grads(param)`
2. 如果是 leaf module，等待所有参数完成
3. 调用 `self.update_hook_state_and_maybe_run_epilogue()`

#### 步骤 4: reduce_independent_p_g_buckets_and_remove_grads

**代码位置:** `deepspeed/runtime/zero/stage3.py:1386`

1. **添加梯度到 bucket** (`__add_grad_to_ipg_bucket`):
   **代码位置:** `deepspeed/runtime/zero/stage3.py:1405`

   - 检查 bucket 容量：如果 `bucket.elements + param.ds_numel > self.reduce_bucket_size`
   - 触发 `__reduce_and_partition_ipg_grads` 归约当前 bucket
   - 将 `param.grad` 复制到 `bucket.buffer` 的连续区域
   - 更新 `bucket.params` 和 `bucket.elements`

2. **归约和分区** (`__reduce_and_partition_ipg_grads`):
   **代码位置:** `deepspeed/runtime/zero/stage3.py:1431`

   - 如果 `contiguous_gradients` 且 `not reduce_scatter`:
     - 调用 `__avg_scatter_contiguous_grads()`
   - 否则：
     - 调用 `__avg_scatter_grads()`
   - 调用 `partition_grads()` 分区梯度到目标缓冲区

#### 步骤 5: __avg_scatter_contiguous_grads

**代码位置:** `deepspeed/runtime/zero/stage3.py:1595`

**通信流程：**
1. **数据类型转换：** 如果 `communication_data_type != dtype`，转换缓冲区
2. **预除：** 如果 `gradient_predivide_factor != 1.0`，除以预除因子
3. **归约：** `buffer_to_reduce.div_(world_sz / sequence_parallel_size)`
4. **All-Reduce：** `dist.all_reduce(buffer_to_reduce, group=self.dp_process_group)`
5. **后乘：** 如果需要，乘以 `gradient_predivide_factor`
6. **类型转换：** 转换回原始数据类型
7. **分区：** 将归约后的缓冲区分割给各个 rank

**通信详情：**
- **GPU 间通信:** All-Reduce 操作，累加所有 rank 的梯度
- **通信数据类型:** 默认 FP16/BF16，可通过 `communication_data_type` 配置
- **NVTX 标记:** 无单独标记，使用函数标记

#### 步骤 6: partition_grads

**代码位置:** `deepspeed/runtime/zero/stage3.py:1728`

**梯度分区和卸载逻辑：**

1. **复制到梯度缓冲区：**
   ```python
   grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
   ```

2. **第一个 micro-batch 处理：**
   ```python
   if self.micro_step_id == 0:  # 不累积，直接复制
       grad_buffer.copy_(grad_partition, non_blocking=True)
       grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
   ```

3. **后续 micro-batch 处理：**
   ```python
   elif get_accelerator().on_accelerator(grad_buffer):
       grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))
   else:
       # CPU offload 情况
       cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
       cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
       grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
   ```

4. **梯度卸载到 CPU (zero-offload 特定)：**
   ```python
   if self.offload_optimizer:
       if self.is_gradient_accumulation_boundary:
           # 计算梯度范数
           self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)

           if self._swappable_optimizer_subgroup(i):
               # 卸载到 pinned memory
               offload_fp32_gradients[i].append(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
               offload_fp32_offsets[i].append(dest_offset)
           else:
               # 直接复制到 CPU fp32 缓冲区
               fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                   0, dest_offset, grad_buffer.numel())
               fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype))
   ```

**通信详情：**
- **GPU-CPU 通信:** DTOH (Device to Host) 数据传输
- **数据转换:** FP16/BF16 -> FP32 (如果需要)
- **Pinned Memory:** 使用 pin_memory 加速传输

#### 步骤 7: independent_gradient_partition_epilogue

**代码位置:** `deepspeed/runtime/zero/stage3.py:1272`

- 对每种数据类型调用 `__reduce_and_partition_ipg_grads()`
- 同步 `reduce_and_partition_stream`
- 重置 `params_already_reduced` 标志
- 设置 `_epilogue_ran_this_backward = True`

---

## 4. Optimizer Step 阶段处理流程

### 4.1 整体流程

```
_pre_step (前置处理)
  └─> 清理 ipg_buckets
  └─> 重置 micro_step_id
  └─> _overflow_check_and_loss_scale_update (溢出检查)
_step (优化器步骤)
  └─> _prepare_sub_group (准备子组)
      └─> _optimizer_states_and_gradient_swap_in (从 CPU swap-in 优化器状态)
  └─> unscale_and_clip_grads (反缩放和梯度裁剪)
  └─> _optimizer_step (执行优化器更新)
  └─> _reassign_or_swap_out_partitioned_parameters (更新参数)
  └─> _release_sub_group (释放子组)
      └─> _optimizer_states_and_gradient_swap_out (swap-out到 CPU)
_post_step (后置处理)
  └─> 持久参数 all-gather
  └─> 清理 CPU 缓冲区
```

### 4.2 详细步骤

#### 步骤 1: zero_grad

**代码位置:** `deepspeed/runtime/zero/stage3.py:2044`

- 设置 `micro_step_id = 0`
- 重置 `_epilogue_ran_this_backward = False`
- 清理 `ipg_buckets` 中的参数 (处理 reentrant checkpointing)
- 记录内存使用

#### 步骤 2: _overflow_check_and_loss_scale_update

**代码位置:** `deepspeed/runtime/zero/stage3.py:2386`

1. 调用 `check_overflow()` 检查梯度是否有 inf/nan
2. 调用 `loss_scaler.update_scale()` 更新 loss scale
3. 如果溢出，调用 `_overflow_clean_up()` 清空梯度

#### 步骤 3: _prepare_sub_group

**代码位置:** `deepspeed/runtime/zero/stage3.py:2260`

对于每个子组 `sub_group_id`：
1. 如果 `_swappable_optimizer_subgroup(sub_group_id)`:
   - 调用 `_optimizer_states_and_gradient_swap_in()`

#### 步骤 4: _optimizer_states_and_gradient_swap_in

**代码位置:** `deepspeed/runtime/zero/stage3.py:2268`

**swap-in 流程：**
1. 从 NVMe/CPU 加载优化器状态到 GPU
2. **CPU-GPU 通信:** HTOD (Host to Device) 数据传输
3. 更新 `fp32_partitioned_groups_flat[i].grad` 指针
4. 使用 pinned memory 加速传输

**代码片段：**
```python
self.optimizer_swapper.swap_in_optimizer_state(
    parameter=self.fp[32_partitioned_groups_flat[sub_group_id],
    async_parameter=self.next_swappable_fp32_partitioned_groups[sub_group_id])
```

#### 步骤 5: unscale_and

_clip_grads

**代码位置:** `deepspeed/runtime/zero/stage3.py:2535`

1. 计算组合缩放因子：`combined_scale = loss_scale / clip_factor`
2. 反缩放梯度：`self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)`

#### 步骤 6: _optimizer_step

**代码位置:** `deepspeed/runtime/zero/stage3.py:1098`

对于 zero-offload：
1. 获取当前设备：`cur_device = self.subgroup_to_device[sub_group_id]`
2. 如果设备是 'cpu'：
   ```python
   self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
   self.optimizer.step()  # DeepSpeedCPUAdam 在 CPU 上执行
   self.optimizer.param_groups[param_group_id]['params'] = []
   ```
3. 如果设备是 GPU：
   ```python
   self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
   self.backup_optimizer.step()  # GPU 优化器
   ```

**计算位置：**
- FP32 参数和梯度在 CPU (DeepSpeedCPUAdam)
- 优化器状态 (exp_avg, exp_avg_sq) 也在 CPU
- 更新后的 FP32 参数在 CPU

#### 步骤 7: _reassign_or_swap_out_partitioned_parameters

**代码位置:** `deepspeed/runtime/zero/stage3.py:2424`

1. 将更新后的 FP32 参数复制回 FP16：
   ```python
   self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
       self.fp32_partitioned_groups_flat[sub_group_id].data)
   ```
2. 调用 `_unflatten_partitioned_parameters()` 还原非扁平化结构

**数据流向：** CPU (FP32) -> GPU (FP16/BF16)

#### 步骤 8: _release_sub_group

**代码位置:** `deepspeed/runtime/zero/stage3.py:2287`

1. 清空梯度：`self.fp32_partitioned_groups_flat[sub_group_id].grad = None`
2. 如果可交换：
   - 调用 `_optimizer_states_and_gradient_swap_out()`

#### 步骤 9: _optimizer_states_and_gradient_swap_out

**代码位置:** `deepspeed/runtime/zero/stage3.py:2318`

**swap-out 流程：**
1. 将优化器状态和参数从 GPU 卸载到 CPU/NVMe
2. **GPU-CPU 通信:** DTOH (Device to Host) 数据传输
3. 更新后，GPU 上的参数可被释放

**代码片段：**
```python
self.optimizer_swapper.swap_out_optimizer_state(
    parameter=self.fp32_partitioned_groups_flat[sub_group_id],
    async_swap=self.next_swappable_fp32_partitioned_groups[sub_group_id] is not None)
```

#### 步骤 10: _post_step

**代码位置:** `deepspeed/runtime/zero/stage3.py:2405`

1. 如果 `offload_optimizer`：调用 `reset_cpu_buffers()`
2. 如果有持久参数：调用 `persistent_parameters[0].all_gather()`
3. 记录内存使用

---

## 5. 数据存储位置和精度总结

### 5.1 Forward 阶段

| Tensor 类型 | 存储位置 | 精度 | 说明 |
|------------|----------|------|------|
| 模型参数 (fp16/bf16) | GPU (临时) | FP16/BF16 | 通过 all-gather 获取，模块计算完成后释放 |
| 激活值 | GPU | FP16/BF16 | 模块计算产生的中间值 |
| 输出张量 | GPU | FP16/BF16 | 传递给下一层 |

### 5.2 Backward 阶段

| Tensor 类型 | 存储位置 | 精度 | 说明 |
|------------|----------|------|------|
| 输入梯度 | GPU | FP16/BF16 | 反向传播产生的梯度 |
| IPG bucket 缓冲区 | GPU | FP16/BF16 | 连续缓冲区，用于批量归约 |
| 梯度分区缓冲区 | GPU -> CPU | FP16/BF16 -> FP32 | 零合后分区存储 |
| 梯度分区缓冲区 (final) | CPU (pinned) | FP32 | `grad_partitions_flat_buffer` |
| 梯度范数缓冲区 | CPU | FP32 | `norm_for_param_grads` |

### 5.3 Optimizer Step 阶段

| Tensor 类型 | 存储位置 | 精度 | 说明 |
|------------|----------|------|------|
| FP32 主参数 | CPU (persistent) | FP32 | `fp32_partitioned_groups_flat` |
| FP32 梯度 | CPU (pinned, 临时) | FP32 | 优化器 step 使用 |
| 优化器状态 (exp_avg) | CPU | FP32 | DeepSpeedCPUAdam 内部维护 |
| 优化器状态 (exp_avg_sq) | CPU | FP32 | DeepSpeedCPUAdam 内部维护 |
| FP16 参数分区 | GPU (临时) | FP16/BF16 | 更新后立即 swap-out |

---

## 6. 关键问题解答

### 6.1 Backward 阶段是否需要 all-gather 获取参数？

**答案：是的，需要。**

**原因：**
在 ZeRO-3 中，参数在 forward 计算完成后会被立即释放。在 `post_sub_module_forward` 中，会调用 `release_sub_module` 将参数状态从 `AVAILABLE` 释放回 `NOT_AVAILABLE`。当 backward 阶段开始时，参数已经不在 GPU 上，因此需要重新通过 all-gather 获取。

**代码证据：**

`deepspeed/runtime/zero/parameter_offload.py:502-509`
```python
@torch.no_grad()
def pre_sub_module_backward_function(self, sub_module):
    param_coordinator = self.get_param_coordinator()
    param_coordinator.trace_prologue(sub_module)
    if param_coordinator.is_record_trace():
        param_coordinator.record_module(sub_module)
    param_coordinator.fetch_sub_module(sub_module, forward=False)  # forward=False 标记
```

### 6.2 是否每个 submodule 计算之后都要 reduce scatter，只保留设备上需要的 gradient？

**答案：是的，每个参数的梯度都会触发 reduce-scatter/all-reduce 操作，并且每个 rank 只保留自己负责的那个梯度分区。**

**原因：**
ZeRO-3 采用梯度分区机制，每个 rank 只负责完整梯度的一部分。当某个参数的梯度在某个 rank 上计算完成后，需要：
1. 将所有 rank 上的梯度部分合并（all-reduce）
2. 计算平均值（除以 world_size）
3. 将平均后的梯度分割（reduce-scatter）
4. 每个 rank 只保留自己负责的那一部分

**代码证据：**

每个参数都会注册 gradient hook：

`deepspeed/runtime/zero/stage3.py:1320-1342`
```python
def wrapper(param):
    def reduce_partition_and_remove_grads(*notneeded):
        self.reduce_ready_partitions_and_remove_grads(param)  # 每个 param 都会触发

    self._grad_acc_hooks.append(register_grad_hook(param, reduce_partition_and_remove_grads))
```

**梯度分区逻辑：**

`deepspeed/runtime/zero/stage3.py:1617-1623`
```python
# 将 all-reduce 后的缓冲区分割给各个 rank
for param in self.ipg_buckets[communication_data_type].params:
    grad = param.grad
    chunk_sz = math.ceil(grad.numel() / world_sz)

    # 每个 rank 只取自己负责的那一部分
    start_offset = grad_offset_in_buffer + min(rank * chunk_sz, grad.numel())
    end_offset = grad_offset_in_buffer + min(rank * chunk_sz + chunk_sz, grad.numel())

    partition = buffer_to_reduce[start_offset:end_offset]
```

---

## 7. GAS=2 时第一个和最后一个 Micro-batch 的区别

### 7.1 Forward 阶段

**第一个 micro-batch (micro_step_id=0):**
- 参数获取：通过 all-gather 从 GPU 收集参数分区
- 参数释放：使用 trace 决定释放策略

**最后一个 micro-batch (micro_step_id=GAS-1):**
- 参数获取：与第一个相同
- 参数释放：与第一个相同
- **区别：** Forward 阶段本身对 micro-batch 无特殊处理，差异主要在 backward

### 7.2 Backward 阶段

**第一个 micro-batch (micro_step_id=0):**
- **梯度处理：** 在 `partition_grads` 中：
  ```python
  if self.micro_step_id == 0:  # 不累积
      grad_buffer.copy_(grad_partition, non_blocking=True)
      grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
  ```
- 直接复制归约后的梯度分区到缓冲区
- **无梯度累积：** 因为是第一个 micro-batch
- **数据流向：** GPU (grad_partition) → GPU/CPU (grad_buffer)

**中间 micro-batch (1 < micro_step_id < GAS-1):**
- **梯度处理：** 在 `partition_grads` 中：
  ```python
  elif get_accelerator().on_accelerator(grad_buffer):
      grad_buffer.add_(grad_partition.to(...))
  else:  # CPU offload
      # 先复制到 GPU，相加，再复制回 CPU
      cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
      cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
      grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
  ```
- **梯度累积：** 将当前 micro-batch 的梯度加到累积缓冲区
- **CPU-GPU 通信：** 每次 add_ 操作都需要 DTOH/HTOH

**最后一个 micro-batch (micro_step_id=GAS-1)：**
- **梯度处理：** 与中间 micro-batch 相同
- **梯度累积：** 累积所有 GAS 个 micro-batch 的梯度
- **梯度范数计算：** 在 `is_gradient_accumulation_boundary` 时：
  ```python
  self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)
  ```
- **梯度卸载：** 将最终累加的梯度卸载到 CPU

### 7.3 Optimizer Step 阶段

**第一个 micro-batch:**
- 不发生（optimizer step 在梯度累积边界调用）

**最后一个 micro-batch:**
- **梯度来源：** CPU 上的累积梯度 (`grad_partitions_flat_buffer`)
- **梯度范数：** 已在 backward 最后一个 micro-batch 时计算
- **优化器更新：** 使用 CPU 上的累积梯度

---

## 8. 通信操作汇总

### 8.1 Forward 阶段

| 通信类型 | 参与 Rank | 方向 | 数据量 | 代码位置 |
|----------|-----------|------|--------|----------|
| All-Gather (参数) | 所有 DP rank | All-to-All | 参数分区大小 | `partitioned_param_coordinator.py:508` |

**NVTX 标记：**
- `FORWARD_FETCH_SUBMIT` - 提交 all-gather 请求
- `FORWARD_ALL_GATHER` - 执行 all-gather 操作
- `FORWARD_PREFETCH_SUBMIT` - 预取操作
- `FORWARD_FETCH_WAIT` - 等待参数就绪

### 8.2 Backward 阶段

| 通信类型 | 参与 Rank | 方向 | 数据量 | 代码位置 |
|----------|-----------|------|--------|----------|
| All-Reduce (梯度) | 所有 DP rank | All-to-All | reduce_bucket_size | `stage3.py:1595` (line 1607) |
| Reduce-Scatter | 所有 DP rank | All-to-All | 梯度分区大小 | `stage3.py:1638` |
| DTOH/HTOH (梯度卸载) | GPU <-> CPU | 双向 | 梯度分区大小 | `stage3.py:1728` |

**NVTX 标记：**
- `__avg_scatter_contiguous_grads` - 梯度归约和分区
- `__avg_scatter_grads` - 梯度归约和分区（非连续）

### 8.3 Optimizer Step 阶段

| 通信类型 | 参与 Rank | 方向 | 数据量 | 代码位置 |
|----------|-----------|------|--------|----------|
| HTOD (状态 swap-in) | CPU -> GPU | 单向 | 优化器状态大小 | `stage3.py:2268` |
| DTOH (状态 swap-out) | GPU -> CPU | 单向 | 优化器状态大小 | `stage3.py:2318` |
| DTOH (参数更新) | GPU -> CPU | 单向 | FP16 参数大小 | `stage3.py:2424` |

**NVTX 标记：**
- `_optimizer_states_and_gradient_swap_in` - swap-in 操作
- `_optimizer_states_and_gradient_swap_out` - swap-out 操作

---

## 9. NVTX 标记完整列表

### Forward 阶段
- `pre_sub_module_forward_function` - `parameter_offload.py:464`
- `_fetch_sub_module_impl` - `partitioned_param_coordinator.py:292`
- `__all_gather_params_` - `partitioned_param_coordinator.py:508`
- `post_sub_module_forward_function` - `parameter_offload.py:484`
- `_release_param` - `partitioned_param_coordinator.py:468`

### Backward 阶段
- `pre_sub_module_backward_function` - `parameter_offload.py:502`
- `reduce_partition_and_remove_grads` - `stage3.py:1326` (装饰器内的函数名)
- `reduce_independent_p_g_buckets_and_remove_grads` - `stage3.py:1386`
- `__add_grad_to_ipg_bucket` - `stage3.py:1405`
- `__reduce_and_partition_ipg_grads` - `stage3.py:1431`
- `__avg_scatter_contiguous_grads` - `stage3.py:1595`
- `__avg_scatter_grads` - `stage3.py:1638`
- `partition_grads` - `stage3.py:1728`
- `independent_gradient_partition_epilogue` - `stage3.py:1272`
- `post_sub_module_backward_function` - `parameter_offload.py:516`

### Optimizer Step 阶段
- `zero_grad` - `stage3.py:2044`
- `_overflow_check_and_loss_scale_update` - `stage3.py:2386`
- `_prepare_sub_group` - `stage3.py:2260`
- `_optimizer_states_and_gradient_swap_in` - `stage3.py:2268`
- `unscale_and_clip_grads` - `stage3.py:2535`
- `_optimizer_step` - `stage3.py:1098`
- `_reassign_or_swap_out_partitioned_parameters` - `stage3.py:2424`
- `_release_sub_group` - `stage3.py:2287`
- `_optimizer_states_and_gradient_swap_out` - `stage3.py:2318`
- `_post_step` - `stage3.py:2405`

---

## 10. 关键数据结构

### 10.1 grad_partitions_flat_buffer

**定义位置:** `deepspeed/runtime/zero/stage3.py:632`

```python
self.grad_partitions_flat_buffer: Tensor = torch.zeros(
    sum(p.partition_numel() for p in all_params),
    dtype=self.gradient_accumulation_dtype,  # 通常是 FP32
    device=self.device  # CPU if offload_optimizer
)
```

**作用：**
- 存储所有参数的梯度分区
- 在 zero-offload 模式下位于 CPU (pinned memory)
- 用于梯度累积和传递给优化器

### 10.2 fp32_partitioned_groups_flat

**定义位置:** `deepspeed/runtime/zero/stage3.py:392`

```python
self.fp32_partitioned_groups_flat = []
# 每个 sub_group 一个 FP32 张量
```

**作用：**
- 存储每个子组的 FP32 主参数
- 在 zero-offload 模式下位于 CPU
- 优化器步骤时更新

### 10.3 norm_for_param_grads

**定义位置:** `deepspeed/runtime/zero/stage3.py:470`

```python
self.norm_for_param_grads = {}
# key: param_id, value: FP32 梯度范数
```

**作用：**
- 存储每个参数的梯度范数
- 用于梯度裁剪
- 在梯度累积边界计算

---

## 11. 性能优化点

1. **Pinned Memory:** CPU-GPU 传输使用 pin_memory，加速数据传输
2. **Pipelined Optimizer Swapping:** 支持流水线式的优化器状态交换
3. **Gradient Bucketing:** 将多个小梯度合并为一个大缓冲区，减少通信次数
4. **Parameter Prefetching:** 基于 trace 预取即将使用的参数
5. **Contiguous Gradients:** 使用连续缓冲区存储梯度，提高访存效率
6. **Asynchronous Communication:** 使用独立的 CUDA stream 进行通信

---

## 12. 总结

DeepSpeed Zero-Offload (ZeRO-3 + offload_optimizer) 的核心特点是：

1. **参数分区：** 模型参数被分区到所有 GPU，通过 all-gather 动态获取
2. **梯度卸载：** 梯度被分区并卸载到 CPU，减少 GPU 内存占用
3. **优化器卸载：** 优化器状态和主参数存储在 CPU，使用 DeepSpeedCPUAdam
4. **梯度累积：** 在 CPU 上累积多个 micro-batch 的梯度
5. **按需加载：** 优化器 step 时按子组 swap-in 优化器状态

这种配置适合：
- 模型参数量远超 GPU 内存
- CPU 内存充足
- 可以容忍一定的通信开销
