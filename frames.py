import torch
import torch.nn as nn
import random
from typing import List, Dict, Any, Tuple, Optional, Union, cast, Callable
import heapq
from collections import defaultdict
from .funcs import interleave_lists
from .special_tokens import special_token_config


class ModalityBlock:
    def __init__(
        self,
        modal_type: str,  # 模态类型，例如"text", "audio", "image"
        units: torch.Tensor,  # 可能为tokens(B, N) 也可能为潜空间(B, N, C)
        masks: Optional[torch.Tensor] = None,  # 用来计算loss的mask(B, N)
        attention_masks: Optional[torch.Tensor] = None,  # 用来遮罩注意力的mask(B, N)
        mixing_mode: str = "condition",  # 混合模式
        condition_level: int = 0,  # 如果为condition混合模式，该块的前后位置
    ):
        assert mixing_mode in ["condition", "dynamic"]
        self.modal_type = modal_type
        self.units = units
        self.mixing_mode = mixing_mode
        self.condition_level = condition_level

        # 确保units不为空
        if units.shape[1] == 0:
            raise ValueError("ModalityBlock units cannot be empty.")

        B, N = units.shape[0], units.shape[1]

        device = units.device
        if masks is None:
            self.masks = torch.ones((B, N), dtype=torch.float32, device=device)
        else:
            self.masks = masks

        if attention_masks is None:
            self.attention_masks = torch.ones(
                (B, N), dtype=torch.bfloat16, device=device
            )
        else:
            self.attention_masks = attention_masks

        # 为特殊token添加属性
        if mixing_mode == "dynamic":
            self.dynamic_special_token_id = getattr(
                special_token_config, f"{modal_type}_dynamic"
            )
        elif mixing_mode == "condition":
            self.condition_special_token_prefix = getattr(
                special_token_config, f"{modal_type}_condition", []
            )
            self.condition_special_token_postfix = (
                getattr(special_token_config, f"end_of_modal")
                if modal_type != "text"
                else getattr(special_token_config, f"end_of_text")
            )

    def __len__(self):
        return self.units.shape[1]

    def __repr__(self):
        return (
            f"ModalityBlock(modal='{self.modal_type}', mode='{self.mixing_mode}', "
            f"level={self.condition_level}, shape={self.units.shape}')"
        )


class ModalityLatentMixingSequence:
    def __init__(
        self,
        latents: torch.Tensor,  # (B, T, C) 混合后的模态潜空间
        modals_per_token: List[str],  # 每个N维度上潜空间单元的对应模态
        reconstruction_metadata: Dict[str, Any],  # 用来重构回Blocks的元信息
        masks: Optional[torch.Tensor] = None,  # (B, T) # 序列对应masks
        attention_masks: Optional[
            torch.Tensor
        ] = None,  # (B, T) # 序列对应attention masks
    ):
        self.latents = latents
        self.modals_per_token = modals_per_token
        self.masks = masks
        self.attention_masks = attention_masks
        self.reconstruction_metadata = reconstruction_metadata

    def rearrange_frame(self) -> "ModalityLatentFrame":
        """
        该函数用来将ModalityLatentFrame.arrange_frame得到的本对象重构回原ModalityLatentFrame
        """
        # 从元数据中解包信息
        block_specs = self.reconstruction_metadata["block_specs"]
        condition_order = self.reconstruction_metadata["condition_order"]
        dynamic_indices = self.reconstruction_metadata["dynamic_indices"]

        B, _, C = self.latents.shape
        device = self.latents.device

        reconstructed_blocks = [None] * len(block_specs)

        # --- 1. 重建 Condition Blocks ---

        # 计算 condition 部分的总长度
        total_condition_len = sum(block_specs[i]["length"] for i in condition_order)

        # 切分出 condition 和 dynamic 两大部分
        if total_condition_len > 0:
            condition_latents = self.latents[:, :total_condition_len, :]
            condition_masks = self.masks[:, :total_condition_len]
            condition_att_masks = self.attention_masks[:, :total_condition_len]

        dynamic_latents = self.latents[:, total_condition_len:, :]
        dynamic_masks = self.masks[:, total_condition_len:]
        dynamic_att_masks = self.attention_masks[:, total_condition_len:]

        # 按顺序重建 condition 块
        current_pos = 0
        for block_idx in condition_order:
            spec = block_specs[block_idx]
            length = spec["length"]

            block_units = condition_latents[:, current_pos : current_pos + length, :]
            block_masks = condition_masks[:, current_pos : current_pos + length]
            block_att_masks = condition_att_masks[:, current_pos : current_pos + length]

            reconstructed_blocks[block_idx] = ModalityBlock(
                modal_type=spec["modal_type"],
                units=block_units,
                masks=block_masks,
                attention_masks=block_att_masks,
                mixing_mode=spec["mixing_mode"],
                condition_level=spec["condition_level"],
            )
            current_pos += length

        # --- 2. 重建 Dynamic Blocks ---
        if dynamic_indices:
            dynamic_blocks_to_process = [block_specs[i] for i in dynamic_indices]

            # 创建占位符列表以模拟交错过程，从而得到逆置换映射
            placeholder_chunks_by_block = []
            for i, spec in enumerate(dynamic_blocks_to_process):
                block_chunks = []
                # 使用spec中的special_token_indices重建分块逻辑
                special_indices_set = set(spec["special_token_indices"])
                k = 0
                while k < spec["length"]:
                    if k in special_indices_set:
                        # (原始 dynamic 列表中的索引, 在块内的起始位置, 长度)
                        block_chunks.append((i, k, 2))
                        k += 2
                    else:
                        block_chunks.append((i, k, 1))
                        k += 1
                placeholder_chunks_by_block.append(block_chunks)

            # 运行交错算法得到交错后的顺序
            interleaved_placeholder = interleave_lists(*placeholder_chunks_by_block)

            # 初始化用于存放重建后数据的列表
            temp_dynamic_units = [
                torch.empty(
                    (B, spec["length"], C), device=device, dtype=self.latents.dtype
                )
                for spec in dynamic_blocks_to_process
            ]
            temp_dynamic_masks = [
                torch.empty((B, spec["length"]), device=device, dtype=self.masks.dtype)
                for spec in dynamic_blocks_to_process
            ]
            temp_dynamic_att_masks = [
                torch.empty(
                    (B, spec["length"]), device=device, dtype=self.attention_masks.dtype
                )
                for spec in dynamic_blocks_to_process
            ]

            # 根据交错顺序，将数据“分发”回各自的块中
            current_pos = 0
            for block_info in interleaved_placeholder:
                block_list_idx, start_idx, chunk_len = block_info

                # 从混合序列中切片
                unit_chunk = dynamic_latents[
                    :, current_pos : current_pos + chunk_len, :
                ]
                mask_chunk = dynamic_masks[:, current_pos : current_pos + chunk_len]
                att_mask_chunk = dynamic_att_masks[
                    :, current_pos : current_pos + chunk_len
                ]

                # 放入对应的临时张量中
                temp_dynamic_units[block_list_idx][
                    :, start_idx : start_idx + chunk_len, :
                ] = unit_chunk
                temp_dynamic_masks[block_list_idx][
                    :, start_idx : start_idx + chunk_len
                ] = mask_chunk
                temp_dynamic_att_masks[block_list_idx][
                    :, start_idx : start_idx + chunk_len
                ] = att_mask_chunk

                current_pos += chunk_len

            # 创建 dynamic 的 ModalityBlock
            for i, original_block_idx in enumerate(dynamic_indices):
                spec = block_specs[original_block_idx]
                reconstructed_blocks[original_block_idx] = ModalityBlock(
                    modal_type=spec["modal_type"],
                    units=temp_dynamic_units[i],
                    masks=temp_dynamic_masks[i],
                    attention_masks=temp_dynamic_att_masks[i],
                    mixing_mode=spec["mixing_mode"],
                    condition_level=spec["condition_level"],
                )

        return ModalityLatentFrame(blocks=reconstructed_blocks)


class ModalityFrame:
    def __init__(self, blocks: List[ModalityBlock]):
        self.blocks = blocks

    @property
    def batch_size(self):
        return self.blocks[0].units.shape[0]

    def shuffle(self):
        """
        混洗blocks，改变不同模态数据出现的顺序
        """
        random.shuffle(self.blocks)

    def _apply_by_key_parallel(
        self,
        module_dict: Dict[str, nn.Module],
        key_func: Callable[[ModalityBlock], str],
    ) -> "ModalityFrame":
        """
        一个通用的并行处理辅助函数。

        它根据 `key_func` 的结果对块进行分组，将同一组的单元拼接后
        通过对应的模块进行一次性处理，然后重组回 ModalityFrame。

        Args:
            module_dict (Dict[str, nn.Module]): 模块字典。
            key_func (Callable[[ModalityBlock], str]): 一个函数，接收一个块并返回一个用于分组的字符串键。

        Returns:
            ModalityFrame: 处理后的新 ModalityFrame。
        """
        # 1. 打包 (Pack): 分组并记录元数据
        grouped_units = defaultdict(list)
        grouped_metadata = defaultdict(list)
        unprocessed_blocks = []

        for i, block in enumerate(self.blocks):
            # 跳过空块的处理
            if len(block) == 0:
                unprocessed_blocks.append((i, block))
                continue

            key = key_func(block)
            if key in module_dict:
                grouped_units[key].append(block.units)
                # 元数据包含：原始索引，长度，以及整个原始块（用于复制其他属性）
                grouped_metadata[key].append(
                    {"original_index": i, "length": len(block), "original_block": block}
                )
            else:
                # 如果没有对应的模块，则将该块视为未处理
                unprocessed_blocks.append((i, block))

        # 准备一个新列表用于重构
        new_blocks = [None] * len(self.blocks)

        # 2. 处理 (Process): 对每个分组进行并行计算
        for key, units_list in grouped_units.items():
            # 拼接
            concatenated_units = torch.cat(units_list, dim=1)
            module = module_dict[key]

            # 处理
            processed_units = module(concatenated_units)

            if (
                processed_units.shape[0] != concatenated_units.shape[0]
                or processed_units.shape[1] != concatenated_units.shape[1]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed Batch or Sequence dimension! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            # 3. 解包 (Unpack): 切分并重构
            metadata_list = grouped_metadata[key]
            lengths = [meta["length"] for meta in metadata_list]
            split_processed_units = torch.split(processed_units, lengths, dim=1)

            for i, unit_chunk in enumerate(split_processed_units):
                meta = metadata_list[i]
                original_block = meta["original_block"]
                original_index = meta["original_index"]

                # 创建新块
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=unit_chunk,
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                new_blocks[original_index] = new_block

        # 将未处理的块放回原位
        for index, block in unprocessed_blocks:
            new_blocks[index] = block

        # 确认所有位置都已填充
        if any(b is None for b in new_blocks):
            raise RuntimeError("Failed to reconstruct all blocks in ModalityFrame.")

        return ModalityFrame(blocks=new_blocks)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityFrame":
        """
        按模态类型对块进行并行处理。

        此方法将相同模态的块拼接在一起，通过相应的模块进行一次性高效处理，
        然后再重构回原始的帧结构。如果某个模态没有对应的模块，其块将保持不变。
        """
        return self._apply_by_key_parallel(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityFrame":
        """
        按（模态类型 + 混合模式）对块进行并行处理。

        此方法将具有相同模态和混合模式的块拼接在一起，进行高效处理。
        """
        return self._apply_by_key_parallel(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    def apply_module_parallel(self, module: nn.Module) -> "ModalityFrame":
        """
        将该帧内所有模态块的潜空间拼接成一个长序列，通过一个torch模块进行并行处理，
        然后重组回一个新的 ModalityFrame。

        此方法假设 'module' 的操作会保持序列长度不变。

        Args:
            module (nn.Module): 一个PyTorch模块，例如一个Transformer层，
                                它将接收一个 (B, N_total, C) 的张量。

        Returns:
            ModalityFrame: 一个新的 ModalityFrame 实例，其中每个块的
                                'units' 都被模块的输出所替换。
        """
        if not self.blocks:
            raise ValueError("Cannot apply module to an empty ModalityFrame.")

        all_units = []
        lengths = []
        original_blocks = []
        for block in self.blocks:
            if len(block) > 0:
                all_units.append(block.units)
                lengths.append(len(block))
                original_blocks.append(block)

        assert len(all_units) > 0, "no units in the frame."

        concatenated_units = torch.cat(all_units, dim=1)

        # 2. 处理 (Process)
        processed_units = module(concatenated_units)

        # 验证模块是否保持了序列长度
        if processed_units.shape[1] != concatenated_units.shape[1]:
            raise ValueError(
                f"The provided module changed the sequence length! "
                f"Input length: {concatenated_units.shape[1]}, "
                f"Output length: {processed_units.shape[1]}"
            )

        # 3. 解包 (Unpack)
        split_processed_units = torch.split(processed_units, lengths, dim=1)

        new_blocks = []
        processed_idx = 0
        for original_block in self.blocks:
            # 使用处理后的单元和原始块的元数据创建新块
            new_block = ModalityBlock(
                modal_type=original_block.modal_type,
                units=split_processed_units[processed_idx],
                masks=original_block.masks,
                attention_masks=original_block.attention_masks,
                mixing_mode=original_block.mixing_mode,
                condition_level=original_block.condition_level,
            )
            processed_idx += 1
            new_blocks.append(new_block)

        return ModalityFrame(blocks=new_blocks)

    def embed(
        self, embedding_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        """
        使用模态特定的嵌入层将token转换为潜空间表示。
        """
        # 调用优化后的并行方法
        processed_frame = self.apply_by_modal(embedding_dict)

        # 将结果封装在 ModalityLatentFrame 中
        return ModalityLatentFrame(blocks=processed_frame.blocks)

    def calc_loss(
        self,
        modality_loss_func_dict: Dict[str, callable],
        origin_frame: "ModalityFrame",
    ) -> Dict[str, torch.Tensor]:
        """可以进一步并行化"""
        if not isinstance(origin_frame, ModalityFrame):
            raise TypeError(
                f"origin_frame must be of type ModalityFrame, but got {type(origin_frame)}"
            )

        if len(self.blocks) != len(origin_frame.blocks):
            raise ValueError(
                "Frame structure mismatch: number of blocks differs "
                f"({len(self.blocks)} vs {len(origin_frame.blocks)})."
            )

        # Use defaultdict to aggregate losses if a modality appears multiple times
        aggregated_losses = defaultdict(list)

        for pred_block, target_block in zip(self.blocks, origin_frame.blocks):
            modal_type = pred_block.modal_type

            # Check for structural consistency
            if modal_type != target_block.modal_type:
                raise ValueError(
                    f"Frame structure mismatch: modal types at the same position differ."
                )

            # If a loss function is defined for this modality, calculate the loss
            if modal_type in modality_loss_func_dict:
                loss_func = modality_loss_func_dict[modal_type]

                # The loss mask comes from the prediction block, as it defines what should be predicted
                loss_mask = pred_block.masks
                bool_mask = loss_mask.bool()

                # Skip if there are no tokens to calculate loss on for this block
                if not bool_mask.any():
                    continue

                # Flatten the prediction and target tensors using the mask
                masked_preds = pred_block.units[bool_mask]
                masked_targets = target_block.units[bool_mask]

                # Calculate and store the loss
                loss = loss_func(masked_preds, masked_targets)
                aggregated_losses[modal_type].append(loss)

        # Average the losses for each modality
        final_losses = {
            modal: torch.mean(torch.stack(loss_list))
            for modal, loss_list in aggregated_losses.items()
        }

        return final_losses


class ModalityLatentFrame:
    def __init__(self, blocks: List[ModalityBlock]):
        self.blocks = blocks

    def _apply_by_key_parallel(
        self,
        module_dict: Dict[str, nn.Module],
        key_func: Callable[[ModalityBlock], str],
    ) -> "ModalityLatentFrame":
        """
        一个通用的并行处理辅助函数。
        与 ModalityFrame 中的版本类似，但返回 ModalityLatentFrame。
        """
        # 1. 打包 (Pack): 分组并记录元数据
        grouped_units = defaultdict(list)
        grouped_metadata = defaultdict(list)
        unprocessed_blocks_with_indices = []

        for i, block in enumerate(self.blocks):
            # 跳过空块的处理
            if len(block) == 0:
                unprocessed_blocks_with_indices.append((i, block))
                continue

            key = key_func(block)
            if key in module_dict:
                grouped_units[key].append(block.units)
                # 元数据包含：原始索引，长度，以及整个原始块（用于复制其他属性）
                grouped_metadata[key].append(
                    {"original_index": i, "length": len(block), "original_block": block}
                )
            else:
                # 如果没有对应的模块，则将该块视为未处理
                unprocessed_blocks_with_indices.append((i, block))

        # 准备一个新列表用于重构
        new_blocks = [None] * len(self.blocks)

        # 2. 处理 (Process): 对每个分组进行并行计算
        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=1)
            module = module_dict[key]
            processed_units = module(concatenated_units)

            if (
                processed_units.shape[0] != concatenated_units.shape[0]
                or processed_units.shape[1] != concatenated_units.shape[1]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed Batch or Sequence dimension! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            # 3. 解包 (Unpack): 切分并重构
            metadata_list = grouped_metadata[key]
            lengths = [meta["length"] for meta in metadata_list]
            split_processed_units = torch.split(processed_units, lengths, dim=1)

            for i, unit_chunk in enumerate(split_processed_units):
                meta = metadata_list[i]
                original_block = meta["original_block"]
                original_index = meta["original_index"]

                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=unit_chunk,
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                new_blocks[original_index] = new_block

        # 将未处理的块和空块放回原位
        for index, block in unprocessed_blocks_with_indices:
            new_blocks[index] = block

        if any(b is None for b in new_blocks):
            raise RuntimeError(
                "Failed to reconstruct all blocks in ModalityLatentFrame."
            )

        return ModalityLatentFrame(blocks=new_blocks)

    def add_condition_special_tokens(
        self,
        embedding: torch.nn.Embedding,
    ):
        for block in self.blocks:
            # 跳过空块
            if len(block) == 0:
                continue
            if block.mixing_mode == "condition":
                B, N, C_block = block.units.shape
                device = block.units.device

                prefix_idx_tensor = torch.tensor(
                    [block.condition_special_token_prefix],
                    dtype=torch.long,
                    device=device,
                )

                postfix_idx_tensor = torch.tensor(
                    [block.condition_special_token_postfix],
                    dtype=torch.long,
                    device=device,
                )

                if embedding.embedding_dim != C_block:
                    raise ValueError(
                        f"Special token embedding dimension ({embedding.embedding_dim}) "
                        f"does not match block's channel dimension ({C_block}) for modal '{block.modal_type}'."
                    )

                prefix_latent = embedding(prefix_idx_tensor)
                postfix_latent = embedding(postfix_idx_tensor)
                prefix_len = len(block.condition_special_token_prefix)
                postfix_len = len(block.condition_special_token_postfix)

                prefix_latent = prefix_latent.expand(B, prefix_len, C_block)
                postfix_latent = postfix_latent.expand(B, postfix_len, C_block)
                block.units = torch.cat(
                    [prefix_latent, block.units, postfix_latent], dim=1
                )
                # response 是唯一指定模型可以自行生成并终结的special token，限串行（alpha）数据模式
                mask_func = (
                    torch.zeros if block.modal_type != "response" else torch.ones
                )
                block.masks = torch.cat(
                    [
                        mask_func((B, prefix_len), dtype=torch.float32, device=device),
                        block.masks,
                        mask_func((B, postfix_len), dtype=torch.float32, device=device),
                    ],
                    dim=1,
                )
                block.attention_masks = torch.cat(
                    [
                        torch.ones(
                            (B, prefix_len), dtype=block.units.dtype, device=device
                        ),
                        block.attention_masks,
                        torch.ones(
                            (B, postfix_len), dtype=block.units.dtype, device=device
                        ),
                    ],
                    dim=1,
                )
        return self

    # def add_random_dynamic_special_tokens(
    #     self,
    #     embedding: torch.nn.Embedding,
    #     prob: float = 0.33,
    #     special_token_metadata: Optional[Dict[int, List[int]]] = None,
    # ) -> Tuple["ModalityLatentFrame", Dict[int, List[int]]]:
    #     """
    #     向 dynamic 模态块中添加特殊 token。
    #     此方法有两种模式：
    #     1. 随机模式 (special_token_metadata=None): 以 `prob` 的概率在 dynamic 块中随机
    #        插入特殊 token，并返回记录了插入位置的元数据。
    #     2. 确定性模式 (special_token_metadata is not None): 根据提供的元数据，在
    #        指定的位置插入特殊 token。

    #     Args:
    #         embedding (torch.nn.Embedding): 用于生成特殊 token 潜空间的 embedding 层。
    #         prob (float): 在随机模式下，每个 dynamic 块被选中插入 token 的概率。
    #         special_token_metadata (Optional[Dict[int, List[int]]]):
    #             一个字典，key 为块索引，value 为该块内要插入 token 的位置列表。
    #             如果提供此参数，将忽略 `prob` 并进入确定性模式。

    #     Returns:
    #         Tuple[ModalityLatentFrame, Dict[int, List[int]]]:
    #         - 第一个元素是修改后的 self。
    #         - 第二个元素是生成或传入的 special_token_metadata。
    #     """
    #     generated_metadata = defaultdict(list)

    #     if special_token_metadata is not None:
    #         # --- 确定性模式 ---
    #         # 按块索引遍历元数据
    #         for block_idx, insert_indices in special_token_metadata.items():
    #             if not insert_indices:
    #                 continue

    #             block = self.blocks[block_idx]
    #             B, N, C_block = block.units.shape
    #             device = block.units.device

    #             special_token_id_tensor = torch.tensor(
    #                 [block.dynamic_special_token_id], dtype=torch.long, device=device
    #             )
    #             special_latent = embedding(special_token_id_tensor)
    #             special_latent_batch = special_latent.expand(B, 1, C_block)

    #             # 关键：必须从后向前插入，否则前面的插入会改变后面位置的索引
    #             for insert_idx in sorted(insert_indices, reverse=True):
    #                 # 更新 units
    #                 block.units = torch.cat(
    #                     [
    #                         block.units[:, :insert_idx, :],
    #                         special_latent_batch,
    #                         block.units[:, insert_idx:, :],
    #                     ],
    #                     dim=1,
    #                 )
    #                 # 更新 masks
    #                 special_mask_val = torch.zeros(
    #                     (B, 1), dtype=torch.float32, device=device
    #                 )
    #                 block.masks = torch.cat(
    #                     [
    #                         block.masks[:, :insert_idx],
    #                         special_mask_val,
    #                         block.masks[:, insert_idx:],
    #                     ],
    #                     dim=1,
    #                 )
    #                 # 更新 attention_masks
    #                 special_att_mask_val = torch.ones(
    #                     (B, 1), dtype=block.units.dtype, device=device
    #                 )
    #                 block.attention_masks = torch.cat(
    #                     [
    #                         block.attention_masks[:, :insert_idx],
    #                         special_att_mask_val,
    #                         block.attention_masks[:, insert_idx:],
    #                     ],
    #                     dim=1,
    #                 )
    #         return self, special_token_metadata

    #     else:
    #         # --- 随机模式 ---
    #         for block_idx, block in enumerate(self.blocks):
    #             # 跳过空块或非 dynamic 块
    #             if len(block) == 0 or block.mixing_mode != "dynamic":
    #                 continue

    #             if random.random() < prob:
    #                 B, N, C_block = block.units.shape
    #                 device = block.units.device

    #                 special_token_id_tensor = torch.tensor(
    #                     [block.dynamic_special_token_id],
    #                     dtype=torch.long,
    #                     device=device,
    #                 )
    #                 special_latent = embedding(special_token_id_tensor)
    #                 special_latent_batch = special_latent.expand(B, 1, C_block)

    #                 # 插入位置保证后面至少还有一个token
    #                 insert_idx = random.randint(0, N - 1)

    #                 # 记录插入位置
    #                 generated_metadata[block_idx].append(insert_idx)

    #                 # 更新 units
    #                 block.units = torch.cat(
    #                     [
    #                         block.units[:, :insert_idx, :],
    #                         special_latent_batch,
    #                         block.units[:, insert_idx:, :],
    #                     ],
    #                     dim=1,
    #                 )
    #                 # 更新 masks
    #                 special_mask_val = torch.zeros(
    #                     (B, 1), dtype=torch.float32, device=device
    #                 )
    #                 block.masks = torch.cat(
    #                     [
    #                         block.masks[:, :insert_idx],
    #                         special_mask_val,
    #                         block.masks[:, insert_idx:],
    #                     ],
    #                     dim=1,
    #                 )
    #                 # 更新 attention_masks
    #                 special_att_mask_val = torch.ones(
    #                     (B, 1), dtype=block.units.dtype, device=device
    #                 )
    #                 block.attention_masks = torch.cat(
    #                     [
    #                         block.attention_masks[:, :insert_idx],
    #                         special_att_mask_val,
    #                         block.attention_masks[:, insert_idx:],
    #                     ],
    #                     dim=1,
    #                 )
    #         return self, dict(
    #             generated_metadata
    #         )  # Convert defaultdict to dict for cleaner output

    def arrange_frame(self) -> "ModalityLatentMixingSequence":
        if not self.blocks:
            raise ValueError("ModalityLatentFrame is empty. Cannot arrange.")

        # 过滤掉空的 block
        valid_blocks = [block for block in self.blocks if len(block) > 0]
        if not valid_blocks:
            raise ValueError(
                "All blocks in ModalityLatentFrame are empty. Cannot arrange."
            )

        ref_block = valid_blocks[0]
        B, _, C = ref_block.units.shape
        device = ref_block.units.device

        # --- 元数据准备 ---
        block_specs = []
        for i, block in enumerate(self.blocks):
            # 找到所有特殊token的位置 (mask=0)
            special_token_indices = (
                (block.masks[0] == 0).nonzero(as_tuple=True)[0].tolist()
            )
            spec = {
                "original_index": i,
                "modal_type": block.modal_type,
                "mixing_mode": block.mixing_mode,
                "condition_level": block.condition_level,
                "length": len(block),
                "special_token_indices": special_token_indices,
            }
            block_specs.append(spec)

        # --- 1. 分离和排序 Condition Blocks ---
        condition_blocks_with_indices = [
            (i, block)
            for i, block in enumerate(self.blocks)
            if block.mixing_mode == "condition" and len(block) > 0
        ]
        # 按 condition_level 排序，如果 level 相同则随机排序
        condition_blocks_with_indices.sort(
            key=lambda x: (x[1].condition_level, random.random())
        )

        final_condition_latents = []
        final_condition_masks = []
        final_condition_att_masks = []
        modals_per_token_condition = []

        sorted_condition_indices = []

        for original_index, block in condition_blocks_with_indices:
            final_condition_latents.append(block.units)
            final_condition_masks.append(block.masks)
            final_condition_att_masks.append(block.attention_masks)
            # 标记每个token的模态和是否是special token
            for i in range(len(block)):
                is_special = block.masks[0, i] == 0
                modals_per_token_condition.append(
                    f"[{block.modal_type}]" if is_special else block.modal_type
                )
            sorted_condition_indices.append(original_index)

        # --- 2. 准备和交错 Dynamic Blocks ---
        dynamic_blocks_with_indices = [
            (i, block)
            for i, block in enumerate(self.blocks)
            if block.mixing_mode == "dynamic" and len(block) > 0
        ]

        dynamic_chunks_by_block = []  # 每个元素是一个block的所有chunk
        original_dynamic_indices = [i for i, b in dynamic_blocks_with_indices]

        for _, block in dynamic_blocks_with_indices:
            chunks = []
            i = 0
            while i < len(block):
                # 检查是否是特殊token (通过mask==0判断)
                if block.masks[0, i] == 0:
                    # 特殊token和它后面的token合并为一个chunk
                    # 注意：要确保不会越界
                    chunk_len = 2 if i + 1 < len(block) else 1
                    chunks.append(
                        (
                            block.units[:, i : i + chunk_len, :],
                            block.masks[:, i : i + chunk_len],
                            block.attention_masks[:, i : i + chunk_len],
                            f"[{block.modal_type}]",  # 第一个是special
                            block.modal_type,
                        )
                    )
                    i += chunk_len
                else:
                    # 普通token自成一个chunk
                    chunks.append(
                        (
                            block.units[:, i : i + 1, :],
                            block.masks[:, i : i + 1],
                            block.attention_masks[:, i : i + 1],
                            block.modal_type,
                        )
                    )
                    i += 1
            dynamic_chunks_by_block.append(chunks)

        interleaved_chunks = interleave_lists(*dynamic_chunks_by_block)

        final_dynamic_latents = [chunk[0] for chunk in interleaved_chunks]
        final_dynamic_masks = [chunk[1] for chunk in interleaved_chunks]
        final_dynamic_att_masks = [chunk[2] for chunk in interleaved_chunks]
        modals_per_token_dynamic = []
        for chunk in interleaved_chunks:
            modals_per_token_dynamic.extend(chunk[3:])

        # --- 3. 合并所有部分 ---
        all_latents = final_condition_latents + final_dynamic_latents
        all_masks = final_condition_masks + final_dynamic_masks
        all_att_masks = final_condition_att_masks + final_dynamic_att_masks

        final_latents = (
            torch.cat(all_latents, dim=1)
            if all_latents
            else torch.empty(B, 0, C, device=device)
        )
        final_masks = (
            torch.cat(all_masks, dim=1)
            if all_masks
            else torch.empty(B, 0, device=device)
        )
        final_att_masks = (
            torch.cat(all_att_masks, dim=1)
            if all_att_masks
            else torch.empty(B, 0, device=device)
        )

        final_modals_per_token = modals_per_token_condition + modals_per_token_dynamic

        # --- 4. 创建重建元数据 ---
        reconstruction_metadata = {
            "block_specs": block_specs,  # 包含所有原始块信息的列表
            "condition_order": sorted_condition_indices,  # condition块的排列顺序
            "dynamic_indices": original_dynamic_indices,  # dynamic块的原始索引
        }

        return ModalityLatentMixingSequence(
            latents=final_latents,
            masks=final_masks,
            attention_masks=final_att_masks,
            modals_per_token=final_modals_per_token,
            reconstruction_metadata=reconstruction_metadata,
        )

    # def apply_attention_mask(self) -> "ModalityLatentFrame":
    #     """
    #     对每个模态块的 'units' 应用其对应的 'attention_masks'。
    #     这是 special dynamic tokens引导假设下到实现，新的假设下实现还没写好。

    #     此操作是不可变的，返回一个新的 ModalityLatentFrame 实例。
    #     计算公式为: new_units = units * attention_mask.

    #     Returns:
    #         ModalityLatentFrame: 一个新的实例，其 units 已被 attention_mask 调节。
    #     """
    #     new_blocks = []
    #     for block in self.blocks:
    #         # 如果块为空，则直接克隆一个功能相同的空块
    #         if len(block) == 0:
    #             new_blocks.append(
    #                 ModalityBlock(
    #                     modal_type=block.modal_type,
    #                     units=block.units.clone(),
    #                     masks=block.masks.clone(),
    #                     attention_masks=block.attention_masks.clone(),
    #                     mixing_mode=block.mixing_mode,
    #                     condition_level=block.condition_level,
    #                 )
    #             )
    #             continue

    #         # 将 attention_mask (B, N) 扩展到 (B, N, 1) 以便与 units (B, N, C) 进行广播乘法
    #         # .unsqueeze(-1) 等同于 [:, :, None]
    #         mask_expanded = block.attention_masks.unsqueeze(-1).to(
    #             dtype=block.units.dtype
    #         )

    #         # 应用掩码
    #         new_units = block.units.mul(mask_expanded)

    #         # 创建一个新块，保留所有原始元数据，只替换 units
    #         new_masked_block = ModalityBlock(
    #             modal_type=block.modal_type,
    #             units=new_units,
    #             masks=block.masks,
    #             attention_masks=block.attention_masks,
    #             mixing_mode=block.mixing_mode,
    #             condition_level=block.condition_level,
    #         )
    #         new_blocks.append(new_masked_block)

    #     return ModalityLatentFrame(blocks=new_blocks)

    # def shift(
    #     self, shift_states_dict: Dict[str, torch.Tensor]
    # ) -> "ModalityLatentFrame":
    #     """
    #     对每个模态块执行时间混合shift，并返回一个新的ModalityLatentFrame。

    #     该方法不会修改原始的 ModalityLatentFrame，而是返回一个包含计算结果的新实例。
    #     计算公式为: new_x[t] = x[t-1] - x[t]，其中 x[-1] 由 shift_state 提供。


    #     未优化
    #     Returns:
    #         ModalityLatentFrame: 一个新的 ModalityLatentFrame 实例，其 units 是经过 shift 计算后的结果。
    #     """

    #     new_blocks = []
    #     for block in self.blocks:
    #         assert len(block) > 0

    #         x = block.units  # (B, N, C) 潜空间张量
    #         B, N, C = x.shape
    #         kk = f"{block.modal_type}_{block.mixing_mode}"

    #         # 检查 shift_state 的维度是否正确
    #         if shift_states_dict[kk].shape != (B, C):
    #             raise ValueError(
    #                 f"Invalid shape for shift_state for modal '{kk}'. "
    #                 f"Expected ({B}, {C}), but got {shift_states_dict.shape}."
    #             )

    #         # 扩展 shift_state，使其可以和序列拼接
    #         # (B, C) -> (B, 1, C)

    #         shift_expanded = shift_states_dict[kk].unsqueeze(1)

    #         # 创建 x_prev 张量，包含移位状态和向右移一位的 x
    #         # x[:, :-1] 取从第一个到倒数第二个的所有元素，形状为 (B, N-1, C)
    #         # torch.cat 拼接后得到 (B, N, C)
    #         x_prev = torch.cat([shift_expanded, x[:, :-1]], dim=1)

    #         xx = x_prev - x  # (B, N, C)

    #         new_shifted_block = ModalityBlock(
    #             modal_type=block.modal_type,
    #             units=xx,
    #             masks=block.masks,  # 掩码长度不变，直接复用
    #             attention_masks=block.attention_masks,
    #             mixing_mode=block.mixing_mode,
    #             condition_level=block.condition_level,
    #         )
    #         new_blocks.append(new_shifted_block)

    #     return ModalityLatentFrame(blocks=new_blocks)


    def shift(
        self, shift_states_dict: Dict[str, torch.Tensor]
    ) -> "ModalityLatentFrame":
        """
        对每个模态块执行时间混合shift，并返回一个新的ModalityLatentFrame。

        该方法不会修改原始的 ModalityLatentFrame，而是返回一个包含计算结果的新实例。
        计算公式为: new_x[t] = x[t-1] - x[t]，其中 x[-1] 由 shift_state 提供。

        此版本经过并行优化，将所有块的计算合并为一次大的张量操作。

        Returns:
            ModalityLatentFrame: 一个新的 ModalityLatentFrame 实例，其 units 是经过 shift 计算后的结果。
        """
        if not self.blocks:
            return ModalityLatentFrame(blocks=[])

        # --- 1. 数据准备 ---
        # 提取所有 units、shift_states 并记录序列长度
        all_units = []
        all_prev_units_parts = []
        split_lengths = []
        
        # 假设所有 block 的 Batch Size (B) 和 Channel (C) 维度都相同
        # 从第一个 block 获取 B 和 C，用于后续的维度检查
        B, _, C = self.blocks[0].units.shape

        for block in self.blocks:
            x = block.units
            if x.shape[0] != B or x.shape[2] != C:
                raise ValueError("All blocks must have the same batch size and channel dimensions for optimized shift.")
            
            kk = f"{block.modal_type}_{block.mixing_mode}"
            shift_state = shift_states_dict[kk]

            # 维度检查
            if shift_state.shape != (B, C):
                raise ValueError(
                    f"Invalid shape for shift_state for modal '{kk}'. "
                    f"Expected ({B}, {C}), but got {shift_state.shape}."
                )

            # 准备用于拼接的各个部分
            all_units.append(x)
            split_lengths.append(x.shape[1])
            
            # 构造 x_prev 的部分: [shift_state, x[:, :-1]]
            shift_expanded = shift_state.unsqueeze(1) # (B, 1, C)
            x_t_minus_1 = x[:, :-1]                   # (B, N-1, C)
            prev_unit_part = torch.cat([shift_expanded, x_t_minus_1], dim=1)
            all_prev_units_parts.append(prev_unit_part)

        # --- 2. 拼接张量 ---
        # 拼接所有当前的 x
        x_concat = torch.cat(all_units, dim=1)
        # 拼接所有上一时刻的 x_prev
        x_prev_concat = torch.cat(all_prev_units_parts, dim=1)

        # --- 3. 并行计算 ---
        # 执行一次性的、高效的减法操作
        xx_concat = x_prev_concat - x_concat

        # --- 4. 分割结果 ---
        # 使用之前记录的长度将结果分割回对应各个 block 的张量
        split_results = torch.split(xx_concat, split_lengths, dim=1)

        # --- 5. 重建对象 ---
        new_blocks = []
        # 使用 zip 将原始 block 的元数据与新的计算结果配对
        for block, new_units in zip(self.blocks, split_results):
            new_shifted_block = ModalityBlock(
                modal_type=block.modal_type,
                units=new_units,
                masks=block.masks,
                attention_masks=block.attention_masks,
                mixing_mode=block.mixing_mode,
                condition_level=block.condition_level,
            )
            new_blocks.append(new_shifted_block)

        return ModalityLatentFrame(blocks=new_blocks)


    def apply_module_parallel(self, module: nn.Module) -> "ModalityLatentFrame":
        """
        将该帧内所有模态块的潜空间拼接成一个长序列，通过一个torch模块进行并行处理，
        然后重组回一个新的 ModalityLatentFrame。

        此方法假设 'module' 的操作会保持序列长度不变。

        Args:
            module (nn.Module): 一个PyTorch模块，例如一个Transformer层，
                                它将接收一个 (B, N_total, C) 的张量。

        Returns:
            ModalityLatentFrame: 一个新的 ModalityLatentFrame 实例，其中每个块的
                                'units' 都被模块的输出所替换。
        """
        if not self.blocks:
            raise ValueError("Cannot apply module to an empty ModalityLatentFrame.")

        all_units = []
        lengths = []
        original_blocks = []
        for block in self.blocks:
            if len(block) > 0:
                all_units.append(block.units)
                lengths.append(len(block))
                original_blocks.append(block)

        # 如果没有可处理的单元，直接返回
        if not all_units:
            return ModalityLatentFrame(blocks=[b for b in self.blocks])  # 返回一个副本

        concatenated_units = torch.cat(all_units, dim=1)

        # 2. 处理 (Process)
        processed_units = module(concatenated_units)

        # 验证模块是否保持了序列长度
        if processed_units.shape[1] != concatenated_units.shape[1]:
            raise ValueError(
                f"The provided module changed the sequence length! "
                f"Input length: {concatenated_units.shape[1]}, "
                f"Output length: {processed_units.shape[1]}"
            )

        # 3. 解包 (Unpack)
        split_processed_units = torch.split(processed_units, lengths, dim=1)

        new_blocks = []
        processed_idx = 0
        for original_block in self.blocks:
            # 如果原始块是空的，则直接克隆一个空块
            if len(original_block) == 0:
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=original_block.units.clone(),
                    masks=original_block.masks.clone(),
                    attention_masks=original_block.attention_masks.clone(),
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
            else:
                # 使用处理后的单元和原始块的元数据创建新块
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=split_processed_units[processed_idx],
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                processed_idx += 1
            new_blocks.append(new_block)

        return ModalityLatentFrame(blocks=new_blocks)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        """
        按模态类型对块进行并行处理。
        """
        return self._apply_by_key_parallel(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        """
        按（模态类型 + 混合模式）对块进行并行处理。
        """
        return self._apply_by_key_parallel(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    @property
    def ctx_len(self) -> int:
        """
        计算该帧内所有模态块拼接后的总序列长度。
        """
        if not self.blocks:
            raise ValueError(
                "Cannot determine context length from an empty ModalityLatentFrame."
            )
        # 使用生成器表达式和sum()，高效计算总长度
        return sum(len(block) for block in self.blocks)

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the blocks in the frame.
        Assumes all blocks have the same batch size.
        """
        if not self.blocks:
            raise ValueError(
                "Cannot determine batch size from an empty ModalityLatentFrame."
            )
        # 修正：应从 self.blocks 获取信息
        return self.blocks[0].units.shape[0]

    def __add__(self, other: "ModalityLatentFrame") -> "ModalityLatentFrame":
        """
        可以进行并行优化
        Element-wise adds the 'units' of two ModalityLatentFrame objects.
        The structure of both frames (number of blocks, order, and tensor shapes) must be identical.
        Masks and other metadata are taken from the left operand (self).

        Args:
            other (ModalityLatentFrame): The frame to add to the current one.

        Returns:
            ModalityLatentFrame: A new frame containing the result of the addition.
        """
        if not isinstance(other, ModalityLatentFrame):
            return NotImplemented  # Standard practice for __add__

        # --- Structural integrity checks ---
        if len(self.blocks) != len(other.blocks):
            raise ValueError(
                f"Cannot add ModalityLatentFrames: number of blocks differs "
                f"({len(self.blocks)} vs {len(other.blocks)})."
            )

        new_blocks = []
        for i, (self_block, other_block) in enumerate(zip(self.blocks, other.blocks)):
            # Check if corresponding blocks are compatible
            if self_block.modal_type != other_block.modal_type:
                raise ValueError(
                    f"Cannot add blocks at index {i}: modal types do not match "
                    f"('{self_block.modal_type}' vs '{other_block.modal_type}')."
                )
            if self_block.units.shape != other_block.units.shape:
                raise ValueError(
                    f"Cannot add blocks for modal '{self_block.modal_type}' at index {i}: "
                    f"shapes are incompatible ({self_block.units.shape} vs {other_block.units.shape})."
                )

            # --- Perform addition and create new block ---
            new_units = self_block.units + other_block.units

            # Create the new block, carrying over metadata from `self`
            result_block = ModalityBlock(
                modal_type=self_block.modal_type,
                units=new_units,
                masks=self_block.masks,
                attention_masks=self_block.attention_masks,
                mixing_mode=self_block.mixing_mode,
                condition_level=self_block.condition_level,
            )
            new_blocks.append(result_block)

        return ModalityLatentFrame(blocks=new_blocks)


class ListModalityLatentMixingSequence(ModalityLatentMixingSequence):
    """
    一个继承自 ModalityLatentMixingSequence 的特殊版本，专门用于处理
    由 ListModalityLatentFrame arrange_list 方法生成的、可并行重构的序列。

    它共享父类的所有属性（latents, masks 等），但提供了自己的
    `rearrange_list` 方法来还原回 ListModalityLatentFrame。
    """

    def __init__(
        self,
        latents: torch.Tensor,
        modals_per_token: List[str],
        reconstruction_metadata: Dict[str, Any],
        masks: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
    ):
        # 使用父类的构造函数来初始化所有核心属性
        super().__init__(
            latents=latents,
            modals_per_token=modals_per_token,
            reconstruction_metadata=reconstruction_metadata,
            masks=masks,
            attention_masks=attention_masks,
        )

    def rearrange_frame(self) -> "ListModalityLatentFrame":
        """将混合序列并行地还原回 ListModalityLatentFrame。"""
        # 1. 首先，调用父类的 rearrange_frame 方法，将序列重建成包含“超级块”的 ModalityLatentFrame
        # 这是重构的第一步，用于解开 condition 和 dynamic 的交错。
        super_blocks_frame = super().rearrange_frame()

        # 2. 从元数据中获取用于列表级重构的信息
        metadata = self.reconstruction_metadata["list_reconstruction_meta"]
        num_frames = metadata["num_frames"]

        # 3. 将超级块按原始长度切分回各自模态的小块
        split_blocks = defaultdict(list)
        for block in super_blocks_frame.blocks:
            modal_type = block.modal_type
            original_lengths = metadata["original_lengths"].get(
                modal_type
            )  # 使用 .get 防止空块问题
            if not original_lengths:
                continue

            # 按记录的长度切分 units 和 masks
            split_units = list(torch.split(block.units, original_lengths, dim=1))
            split_masks = list(torch.split(block.masks, original_lengths, dim=1))
            split_att_masks = list(
                torch.split(block.attention_masks, original_lengths, dim=1)
            )

            for i in range(len(split_units)):
                # 为每个切分后的块重新创建 ModalityBlock 对象
                split_blocks[modal_type].append(
                    ModalityBlock(
                        modal_type=modal_type,
                        units=split_units[i],
                        masks=split_masks[i],
                        attention_masks=split_att_masks[i],
                        mixing_mode=block.mixing_mode,
                        condition_level=block.condition_level,
                    )
                )

        # 4. 根据原始帧的模态顺序，重新组装成 ListModalityLatentFrame
        rearranged_frames = []
        for i in range(num_frames):
            frame_blocks = []
            for modal_type in metadata["modal_order_in_frame"][i]:
                if split_blocks[modal_type]:
                    frame_blocks.append(split_blocks[modal_type].pop(0))
            rearranged_frames.append(ModalityLatentFrame(blocks=frame_blocks))

        return ListModalityLatentFrame(frames=rearranged_frames)


class ListModalityFrame:
    """
    A container for a list of ModalityFrame objects, providing methods
    for parallel processing of all frames in the list.
    """

    def __init__(self, frames: List[ModalityFrame]):
        self.frames = frames

    @property
    def batch_size(self):
        return self.frames[0].batch_size

    def shuffle_frame(self):
        for frame in self.frames:
            frame.shuffle()

    def _apply_by_key_parallel_list(
        self, module_dict: Dict[str, nn.Module], key_func: Callable[[ModalityBlock], str]
    ) -> "ListModalityFrame":
        """
        一个通用的、跨所有帧的并行处理辅助函数。

        Args:
            module_dict (Dict[str, nn.Module]): 模块字典。
            key_func (Callable[[ModalityBlock], str]): 用于分组的键生成函数。

        Returns:
            ListModalityFrame: 处理后的新 ListModalityFrame。
        """
        if not self.frames:
            return ListModalityFrame(frames=[])

        # 1. 打包 (Pack): 跨所有帧进行分组和元数据记录
        grouped_units = defaultdict(list)
        block_positions = [] # 记录每个块的原始位置和信息

        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                key = key_func(block)
                position_info = {
                    "frame_idx": frame_idx,
                    "block_idx": block_idx,
                    "processed": False,
                    "block": block,
                    "key": key
                }

                if len(block) > 0 and key in module_dict:
                    grouped_units[key].append(block.units)
                    position_info["processed"] = True
                
                block_positions.append(position_info)

        # 2. 处理 (Process)
        processed_unit_chunks = defaultdict(list)
        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=1)
            module = module_dict[key]
            processed_units = module(concatenated_units)

            if (processed_units.shape[0] != concatenated_units.shape[0] or
                processed_units.shape[1] != concatenated_units.shape[1]):
                raise ValueError(
                    f"Module for key '{key}' changed Batch or Sequence dimension! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )
            
            # 按原始长度切分
            lengths = [
                len(pos["block"]) for pos in block_positions 
                if pos["processed"] and pos["key"] == key
            ]
            processed_unit_chunks[key] = list(torch.split(processed_units, lengths, dim=1))

        # 3. 解包 (Unpack)
        reconstruction_data = [[] for _ in self.frames]
        counters = defaultdict(int)

        for pos_info in block_positions:
            frame_idx, block_idx = pos_info["frame_idx"], pos_info["block_idx"]
            original_block = pos_info["block"]

            if pos_info["processed"]:
                key = pos_info["key"]
                chunk_index = counters[key]
                new_unit_chunk = processed_unit_chunks[key][chunk_index]
                
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=new_unit_chunk,
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                reconstruction_data[frame_idx].append((block_idx, new_block))
                counters[key] += 1
            else:
                reconstruction_data[frame_idx].append((block_idx, original_block))

        # 重建
        new_frames = []
        for frame_idx in range(len(self.frames)):
            sorted_blocks = sorted(reconstruction_data[frame_idx], key=lambda x: x[0])
            new_blocks_for_frame = [b for _, b in sorted_blocks]
            new_frames.append(ModalityFrame(blocks=new_blocks_for_frame))

        return ListModalityFrame(frames=new_frames)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityFrame":
        """
        对列表中所有帧的所有块按模态进行一次性并行处理。
        """
        return self._apply_by_key_parallel_list(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityFrame":
        """
        对列表中所有帧的所有块按（模态+混合模式）进行并行处理。
        """
        return self._apply_by_key_parallel_list(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    def embed(
        self, embedding_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        """
        使用模态特定的嵌入层将token转换为潜空间表示。

        此方法通过一次性并行处理所有帧中的所有块来实现高效嵌入。
        """
        # 直接调用已经优化好的、全局并行的 apply_by_modal 方法
        # 它会返回一个处理后的 ListModalityFrame
        processed_list_frame = self.apply_by_modal(embedding_dict)
        
        # 结果 ListModalityFrame 中的每个 Frame 的 block 已经是潜空间表示了，
        # 我们只需要将它们重新包装成 ListModalityLatentFrame。
        
        latent_frames = []
        for frame in processed_list_frame.frames:
            # frame.blocks 已经是处理后的 latent blocks
            latent_frames.append(ModalityLatentFrame(blocks=frame.blocks))
            
        return ListModalityLatentFrame(frames=latent_frames)

    def get_modality_units(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        将所有帧中的单元按模态分组并拼接，同时生成用于重构的元数据。

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
            - 第一个元素是一个字典，键是模态类型 (str)，值是拼接后的 units 张量。
            - 第二个元素是元数据字典，包含重构 ListModalityLatentFrame 所需的所有信息。
        """
        grouped_units = defaultdict(list)
        original_lengths = defaultdict(list)
        modal_order_in_frame = []
        # We need to save all other properties to reconstruct the blocks later
        original_blocks_properties = defaultdict(list)

        # 1. Iterate and group all data
        for frame in self.frames:
            modal_order_in_frame.append([b.modal_type for b in frame.blocks])
            for block in frame.blocks:
                modal_type = block.modal_type
                grouped_units[modal_type].append(block.units)
                original_lengths[modal_type].append(len(block))
                # Store everything except the units
                original_blocks_properties[modal_type].append(
                    {
                        "masks": block.masks,
                        "attention_masks": block.attention_masks,
                        "mixing_mode": block.mixing_mode,
                        "condition_level": block.condition_level,
                    }
                )

        # 2. Concatenate the grouped units
        concatenated_units = {
            modal_type: torch.cat(units_list, dim=1)
            for modal_type, units_list in grouped_units.items()
        }

        # 3. Assemble the metadata for reconstruction
        metadata = {
            "num_frames": len(self.frames),
            "modal_order_in_frame": modal_order_in_frame,
            "original_lengths": original_lengths,
            "original_blocks_properties": original_blocks_properties,
        }

        return concatenated_units, metadata

    def apply_module_parallel(self, module: nn.Module) -> "ListModalityFrame":
        """
        将列表中所有帧的所有模态块的潜空间拼接成一个超长序列，通过一个torch模块
        进行并行处理，然后重组回一个新的 ListModalityFrame。

        此方法假设 'module' 的操作会保持序列长度不变。

        Args:
            module (nn.Module): 一个PyTorch模块，例如一个Transformer层。

        Returns:
            ListModalityFrame: 一个新的 ListModalityFrame 实例，其中
                                    每个块的 'units' 都被模块的输出所替换。
        """
        if not self.frames:
            return self

        all_units = []
        metadata = []

        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                if len(block) > 0:
                    all_units.append(block.units)
                    # 存储足够的信息以便完美重构
                    metadata.append(
                        {
                            "frame_idx": frame_idx,
                            "block_idx": block_idx,
                            "length": len(block),
                            "original_block": block,  # 存储整个块以便于复制元数据
                        }
                    )

        if not all_units:
            # 如果所有块都为空，则返回一个深层副本
            return ListModalityFrame(
                frames=[
                    ModalityFrame(
                        blocks=[
                            ModalityBlock(
                                modal_type=b.modal_type,
                                units=b.units.clone(),
                                masks=b.masks.clone(),
                                attention_masks=b.attention_masks.clone(),
                                mixing_mode=b.mixing_mode,
                                condition_level=b.condition_level,
                            )
                            for b in f.blocks
                        ]
                    )
                    for f in self.frames
                ]
            )

        concatenated_units = torch.cat(all_units, dim=1)

        # 2. 处理 (Process)
        processed_units = module(concatenated_units)

        # 验证模块是否保持了序列长度
        if processed_units.shape[1] != concatenated_units.shape[1]:
            raise ValueError(
                f"The provided module changed the sequence length! "
                f"Input length: {concatenated_units.shape[1]}, "
                f"Output length: {processed_units.shape[1]}"
            )

        # 3. 解包 (Unpack)
        lengths = [meta["length"] for meta in metadata]
        split_processed_units = torch.split(processed_units, lengths, dim=1)

        # 创建一个临时结构来存放将被重组的新块
        # 结构: [ [ (block_idx, new_block), ... ], ... ]
        reconstruction_data = [[] for _ in self.frames]

        # 填充所有处理过的新块
        for i, meta_info in enumerate(metadata):
            original_block = meta_info["original_block"]
            new_block = ModalityBlock(
                modal_type=original_block.modal_type,
                units=split_processed_units[i],
                masks=original_block.masks,
                attention_masks=original_block.attention_masks,
                mixing_mode=original_block.mixing_mode,
                condition_level=original_block.condition_level,
            )
            reconstruction_data[meta_info["frame_idx"]].append(
                (meta_info["block_idx"], new_block)
            )

        # 重建帧和块的完整结构
        new_frames = []
        for frame_idx, frame in enumerate(self.frames):
            new_blocks_for_frame = []

            # 将已处理的块按原始顺序放回
            processed_blocks_map = {
                block_idx: block for block_idx, block in reconstruction_data[frame_idx]
            }

            for block_idx, original_block in enumerate(frame.blocks):
                if block_idx in processed_blocks_map:
                    new_blocks_for_frame.append(processed_blocks_map[block_idx])
                else:
                    # 这个分支处理原始的空块
                    new_empty_block = ModalityBlock(
                        modal_type=original_block.modal_type,
                        units=original_block.units.clone(),
                        masks=original_block.masks.clone(),
                        attention_masks=original_block.attention_masks.clone(),
                        mixing_mode=original_block.mixing_mode,
                        condition_level=original_block.condition_level,
                    )
                    new_blocks_for_frame.append(new_empty_block)

            new_frames.append(ModalityFrame(blocks=new_blocks_for_frame))

        return ListModalityFrame(frames=new_frames)


class ListModalityLatentFrame:
    def __init__(self, frames: List[ModalityLatentFrame]):
        self.frames = frames

    def get_modality_units(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        将所有帧中的单元按模态分组并拼接，同时生成用于重构的元数据。

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
            - 第一个元素是一个字典，键是模态类型 (str)，值是拼接后的 units 张量。
            - 第二个元素是元数据字典，包含重构 ListModalityLatentFrame 所需的所有信息。
        """
        grouped_units = defaultdict(list)
        original_lengths = defaultdict(list)
        modal_order_in_frame = []
        # We need to save all other properties to reconstruct the blocks later
        original_blocks_properties = defaultdict(list)

        # 1. Iterate and group all data
        for frame in self.frames:
            modal_order_in_frame.append([b.modal_type for b in frame.blocks])
            for block in frame.blocks:
                modal_type = block.modal_type
                grouped_units[modal_type].append(block.units)
                original_lengths[modal_type].append(len(block))
                # Store everything except the units
                original_blocks_properties[modal_type].append(
                    {
                        "masks": block.masks,
                        "attention_masks": block.attention_masks,
                        "mixing_mode": block.mixing_mode,
                        "condition_level": block.condition_level,
                    }
                )

        # 2. Concatenate the grouped units
        concatenated_units = {
            modal_type: torch.cat(units_list, dim=1)
            for modal_type, units_list in grouped_units.items()
        }

        # 3. Assemble the metadata for reconstruction
        metadata = {
            "num_frames": len(self.frames),
            "modal_order_in_frame": modal_order_in_frame,
            "original_lengths": original_lengths,
            "original_blocks_properties": original_blocks_properties,
        }

        return concatenated_units, metadata

    @staticmethod
    def reconstruct_from_units(
        units_dict: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> "ListModalityLatentFrame":
        """
        使用拼接后的 units 和元数据，静态地重构一个 ListModalityLatentFrame。

        这允许在外部对 units 进行操作（只要形状不变），然后重新组装。

        Args:
            units_dict (Dict[str, torch.Tensor]): 按模态分组的（可能已修改的）拼接后 units。
            metadata (Dict[str, Any]): 由 get_modality_units 生成的元数据。

        Returns:
            ListModalityLatentFrame: 重构后的对象。
        """
        # Unpack metadata
        num_frames = metadata["num_frames"]
        modal_order_in_frame = metadata["modal_order_in_frame"]
        original_lengths = metadata["original_lengths"]
        original_blocks_properties = metadata["original_blocks_properties"]

        # 1. Split the concatenated tensors back into chunks
        split_blocks = defaultdict(list)
        for modal_type, concatenated_units in units_dict.items():
            lengths = original_lengths[modal_type]
            properties_list = original_blocks_properties[modal_type]

            # Split the units tensor according to the original block lengths
            unit_chunks = list(torch.split(concatenated_units, lengths, dim=1))

            # Recreate each ModalityBlock by pairing the unit chunk with its original properties
            for i, unit_chunk in enumerate(unit_chunks):
                block_properties = properties_list[i]
                recreated_block = ModalityBlock(
                    modal_type=modal_type, units=unit_chunk, **block_properties
                )
                split_blocks[modal_type].append(recreated_block)

        # 2. Reassemble the frames using the reconstructed blocks
        new_frames = []
        for i in range(num_frames):
            frame_blocks = []
            # Use the original modal order for each frame to pull blocks from the queues
            for modal_type in modal_order_in_frame[i]:
                if split_blocks[modal_type]:
                    frame_blocks.append(split_blocks[modal_type].pop(0))
            new_frames.append(ModalityLatentFrame(blocks=frame_blocks))

        return ListModalityLatentFrame(frames=new_frames)

    def add_condition_special_tokens(
        self, embedding: torch.nn.Embedding
    ) -> "ListModalityLatentFrame":
        """
        对列表中所有帧的所有 condition 模态块添加特殊的前缀和后缀 token。

        此操作通过调用每个子帧的 `add_condition_special_tokens` 方法来实现，
        会就地修改每个 frame。

        Args:
            embedding (torch.nn.Embedding): 用于从 token ID 生成潜空间的 embedding 层。

        Returns:
            ListModalityLatentFrame: 返回 self，以支持方法链式调用。
        """
        for frame in self.frames:
            frame.add_condition_special_tokens(embedding)
        return self

    # def add_random_dynamic_special_tokens(
    #     self,
    #     embedding: torch.nn.Embedding,
    #     prob: float = 0.33,
    #     special_token_metadata: Optional[List[Dict[int, List[int]]]] = None,
    # ) -> Tuple["ListModalityLatentFrame", List[Dict[int, List[int]]]]:
    #     """
    #     对列表中的所有 frame 并行地添加特殊 token。
    #     此方法通过委托给 ModalityLatentFrame.add_random_dynamic_special_tokens 实现。

    #     Args:
    #         embedding (torch.nn.Embedding): Embedding层。
    #         prob (float): 随机模式下的概率。
    #         special_token_metadata (Optional[List[Dict[int, List[int]]]]):
    #             一个元数据列表，每个元素对应一个 frame 的元数据。
    #             如果提供此参数，将进入确定性模式。

    #     Returns:
    #         Tuple[ListModalityLatentFrame, List[Dict[int, List[int]]]]:
    #         - 第一个元素是修改后的 self。
    #         - 第二个元素是生成或传入的元数据列表。
    #     """
    #     if special_token_metadata is not None:
    #         # --- 确定性模式 ---
    #         if len(self.frames) != len(special_token_metadata):
    #             raise ValueError(
    #                 f"Number of frames ({len(self.frames)}) does not match "
    #                 f"the length of special_token_metadata ({len(special_token_metadata)})."
    #             )

    #         for frame, meta_for_frame in zip(self.frames, special_token_metadata):
    #             frame.add_random_dynamic_special_tokens(
    #                 embedding=embedding, special_token_metadata=meta_for_frame
    #             )
    #         return self, special_token_metadata
    #     else:
    #         # --- 随机模式 ---
    #         all_metadata = []
    #         for frame in self.frames:
    #             _, meta_for_frame = frame.add_random_dynamic_special_tokens(
    #                 embedding=embedding, prob=prob
    #             )
    #             all_metadata.append(meta_for_frame)
    #         return self, all_metadata

    def shift(
        self, shift_states_dict: Dict[str, torch.Tensor]
    ) -> "ListModalityLatentFrame":
        """
        对列表中的所有帧执行一个连续的时间混合shift操作。

        此方法将列表中的帧视为一个长的时间序列。第一个帧使用外部提供的
        `shift_states_dict` 作为其初始状态。对于后续的每一个帧，其初始
        shift状态是前一个原始帧的最后一个时间步。

        计算公式为: new_x[t] = x[t-1] - x[t]

        Args:
            shift_states_dict (Dict[str, torch.Tensor]): 一个字典，键为
                '{modal_type}_{mixing_mode}'，值为每个模态的初始移位状态张量 (B, C)。

        Returns:
            ListModalityLatentFrame: 一个新的 ListModalityLatentFrame 实例，
                                     其中所有单元都经过了连续的 shift 计算。
        """
        if not self.frames:
            return ListModalityLatentFrame(frames=[])

        new_frames = []
        # 复制初始状态，以避免修改外部字典
        current_shift_states = shift_states_dict.copy()

        for frame in self.frames:
            # 1. 对当前帧应用 shift 操作
            # ModalityLatentFrame.shift 已经实现了对帧内所有块的并行处理
            shifted_frame = frame.shift(current_shift_states)
            new_frames.append(shifted_frame)

            # 2. 准备下一个帧的初始状态
            # 下一个状态是当前 *原始* 帧的最后一个时间步
            next_states = {}
            for block in frame.blocks:
                # 只为非空块提取状态
                if len(block) > 0:
                    kk = f"{block.modal_type}_{block.mixing_mode}"
                    # 提取最后一个时间步 (B, N, C) -> (B, C)
                    last_timestep = block.units[:, -1, :]
                    next_states[kk] = last_timestep

            # 3. 更新状态以用于下一次循环
            current_shift_states = next_states

        return ListModalityLatentFrame(frames=new_frames)



    def arrange_frame(self) -> "ListModalityLatentMixingSequence":
        """将 ListModalityLatentFrame 并行地转换为 ListModalityLatentMixingSequence。"""
        if not self.frames:
            raise ValueError("Cannot arrange an empty list of frames.")

        grouped_blocks = defaultdict(list)
        original_lengths = defaultdict(list)
        modal_order_in_frame = []

        # 将所有帧中相同模态的块分组
        for frame in self.frames:
            modal_order_in_frame.append([b.modal_type for b in frame.blocks])
            for block in frame.blocks:
                grouped_blocks[block.modal_type].append(block)
                original_lengths[block.modal_type].append(len(block))

        # 为每种模态创建一个拼接后的“超级块”
        super_blocks = []
        for modal_type in sorted(grouped_blocks.keys()):
            blocks = grouped_blocks[modal_type]
            ref_block = blocks[0]
            concatenated_units = torch.cat([b.units for b in blocks], dim=1)
            concatenated_masks = torch.cat([b.masks for b in blocks], dim=1)
            concatenated_att_masks = torch.cat(
                [b.attention_masks for b in blocks], dim=1
            )
            super_blocks.append(
                ModalityBlock(
                    modal_type,
                    concatenated_units,
                    concatenated_masks,
                    concatenated_att_masks,
                    ref_block.mixing_mode,
                    ref_block.condition_level,
                )
            )

        # 将“超级块”放入一个临时的 ModalityLatentFrame 中，以便调用 arrange_frame
        temp_frame = ModalityLatentFrame(blocks=super_blocks)

        # 调用 arrange_frame 得到一个标准的混合序列对象
        mixed_sequence_obj = temp_frame.arrange_frame()

        # 在元数据中添加用于列表级重构的额外信息
        mixed_sequence_obj.reconstruction_metadata["list_reconstruction_meta"] = {
            "num_frames": len(self.frames),
            "original_lengths": original_lengths,
            "modal_order_in_frame": modal_order_in_frame,
        }

        # *** --- 改动部分 --- ***
        # 使用混合序列对象的数据来实例化 ListModalityLatentMixingSequence
        return ListModalityLatentMixingSequence(
            latents=mixed_sequence_obj.latents,
            modals_per_token=mixed_sequence_obj.modals_per_token,
            reconstruction_metadata=mixed_sequence_obj.reconstruction_metadata,
            masks=mixed_sequence_obj.masks,
            attention_masks=mixed_sequence_obj.attention_masks,
        )

    def apply_module_parallel(self, module: nn.Module) -> "ListModalityLatentFrame":
        """
        将列表中所有帧的所有模态块的潜空间拼接成一个超长序列，通过一个torch模块
        进行并行处理，然后重组回一个新的 ListModalityLatentFrame。

        此方法假设 'module' 的操作会保持序列长度不变。

        Args:
            module (nn.Module): 一个PyTorch模块，例如一个Transformer层。

        Returns:
            ListModalityLatentFrame: 一个新的 ListModalityLatentFrame 实例，其中
                                    每个块的 'units' 都被模块的输出所替换。
        """
        if not self.frames:
            return self

        # 1. 打包 (Pack)
        all_units = []
        # 使用元数据来记录每个单元块的原始位置和信息
        metadata = []

        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                if len(block) > 0:
                    all_units.append(block.units)
                    # 存储足够的信息以便完美重构
                    metadata.append(
                        {
                            "frame_idx": frame_idx,
                            "block_idx": block_idx,
                            "length": len(block),
                            "original_block": block,  # 存储整个块以便于复制元数据
                        }
                    )

        if not all_units:
            # 如果所有块都为空，则返回一个深层副本
            return ListModalityLatentFrame(
                frames=[
                    ModalityLatentFrame(
                        blocks=[
                            ModalityBlock(
                                modal_type=b.modal_type,
                                units=b.units.clone(),
                                masks=b.masks.clone(),
                                attention_masks=b.attention_masks.clone(),
                                mixing_mode=b.mixing_mode,
                                condition_level=b.condition_level,
                            )
                            for b in f.blocks
                        ]
                    )
                    for f in self.frames
                ]
            )

        concatenated_units = torch.cat(all_units, dim=1)

        # 2. 处理 (Process)
        processed_units = module(concatenated_units)

        # 验证模块是否保持了序列长度
        if processed_units.shape[1] != concatenated_units.shape[1]:
            raise ValueError(
                f"The provided module changed the sequence length! "
                f"Input length: {concatenated_units.shape[1]}, "
                f"Output length: {processed_units.shape[1]}"
            )

        # 3. 解包 (Unpack)
        lengths = [meta["length"] for meta in metadata]
        split_processed_units = torch.split(processed_units, lengths, dim=1)

        # 创建一个临时结构来存放将被重组的新块
        # 结构: [ [ (block_idx, new_block), ... ], ... ]
        reconstruction_data = [[] for _ in self.frames]

        # 填充所有处理过的新块
        for i, meta_info in enumerate(metadata):
            original_block = meta_info["original_block"]
            new_block = ModalityBlock(
                modal_type=original_block.modal_type,
                units=split_processed_units[i],
                masks=original_block.masks,
                attention_masks=original_block.attention_masks,
                mixing_mode=original_block.mixing_mode,
                condition_level=original_block.condition_level,
            )
            reconstruction_data[meta_info["frame_idx"]].append(
                (meta_info["block_idx"], new_block)
            )

        # 重建帧和块的完整结构
        new_frames = []
        for frame_idx, frame in enumerate(self.frames):
            new_blocks_for_frame = []

            # 将已处理的块按原始顺序放回
            processed_blocks_map = {
                block_idx: block for block_idx, block in reconstruction_data[frame_idx]
            }

            for block_idx, original_block in enumerate(frame.blocks):
                if block_idx in processed_blocks_map:
                    new_blocks_for_frame.append(processed_blocks_map[block_idx])
                else:
                    # 这个分支处理原始的空块
                    new_empty_block = ModalityBlock(
                        modal_type=original_block.modal_type,
                        units=original_block.units.clone(),
                        masks=original_block.masks.clone(),
                        attention_masks=original_block.attention_masks.clone(),
                        mixing_mode=original_block.mixing_mode,
                        condition_level=original_block.condition_level,
                    )
                    new_blocks_for_frame.append(new_empty_block)

            new_frames.append(ModalityLatentFrame(blocks=new_blocks_for_frame))

        return ListModalityLatentFrame(frames=new_frames)

    def _apply_by_key_parallel_list(
        self, module_dict: Dict[str, nn.Module], key_func: Callable[[ModalityBlock], str]
    ) -> "ListModalityLatentFrame":
        """
        一个通用的、跨所有帧的并行处理辅助函数。
        """
        if not self.frames:
            return ListModalityLatentFrame(frames=[])

        # 1. 打包 (Pack)
        grouped_units = defaultdict(list)
        block_positions = []

        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                key = key_func(block)
                position_info = {
                    "frame_idx": frame_idx, "block_idx": block_idx,
                    "processed": False, "block": block, "key": key
                }
                if len(block) > 0 and key in module_dict:
                    grouped_units[key].append(block.units)
                    position_info["processed"] = True
                block_positions.append(position_info)

        # 2. 处理 (Process)
        processed_unit_chunks = defaultdict(list)
        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=1)
            processed_units = module_dict[key](concatenated_units)

            if (processed_units.shape[0] != concatenated_units.shape[0] or
                processed_units.shape[1] != concatenated_units.shape[1]):
                raise ValueError(
                    f"Module for key '{key}' changed Batch or Sequence dimension! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )
            
            lengths = [
                len(pos["block"]) for pos in block_positions 
                if pos["processed"] and pos["key"] == key
            ]
            processed_unit_chunks[key] = list(torch.split(processed_units, lengths, dim=1))

        # 3. 解包 (Unpack)
        reconstruction_data = [[] for _ in self.frames]
        counters = defaultdict(int)

        for pos_info in block_positions:
            frame_idx, block_idx = pos_info["frame_idx"], pos_info["block_idx"]
            original_block = pos_info["block"]

            if pos_info["processed"]:
                key = pos_info["key"]
                new_unit_chunk = processed_unit_chunks[key][counters[key]]
                
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type, units=new_unit_chunk,
                    masks=original_block.masks, attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode, condition_level=original_block.condition_level
                )
                reconstruction_data[frame_idx].append((block_idx, new_block))
                counters[key] += 1
            else:
                reconstruction_data[frame_idx].append((block_idx, original_block))

        # 重建
        new_frames = []
        for frame_idx in range(len(self.frames)):
            sorted_blocks = sorted(reconstruction_data[frame_idx], key=lambda x: x[0])
            new_blocks = [b for _, b in sorted_blocks]
            new_frames.append(ModalityLatentFrame(blocks=new_blocks))

        return ListModalityLatentFrame(frames=new_frames)


    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        """
        对列表中所有帧的所有块按模态进行一次性并行处理。
        """
        return self._apply_by_key_parallel_list(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        """
        对列表中所有帧的所有块按（模态+混合模式）进行并行处理。
        """
        return self._apply_by_key_parallel_list(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    def apply_attention_mask(self) -> "ListModalityLatentFrame":
        """
        对列表中所有帧的所有模态块并行地应用 'attention_masks'。
        # 这是 special dynamic tokens引导假设下到实现，新的假设下实现还没写好。
        此操作通过调用每个子帧的 apply_attention_mask 方法来实现。

        Returns:
            ListModalityLatentFrame: 一个新的实例，其所有 units 都已被 attention_mask 调节。
        """
        # 优雅地将操作委托给每个 ModalityLatentFrame 子对象
        new_frames = [frame.apply_attention_mask() for frame in self.frames]
        return ListModalityLatentFrame(frames=new_frames)

    @property
    def ctx_len(self) -> int:
        """
        计算列表中所有帧的所有模态块拼接后的总序列长度。
        """
        if not self.frames:
            return 0
        # 嵌套的生成器表达式，非常高效
        return sum(len(block) for frame in self.frames for block in frame.blocks)

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the latent frames in the list.
        Assumes all frames have the same batch size.
        """
        if not self.frames:
            raise ValueError(
                "Cannot determine batch size from an empty ListModalityLatentFrame."
            )
        # Delegates to the batch_size property of the first frame
        return self.frames[0].batch_size

    def __add__(self, other: "ListModalityLatentFrame") -> "ListModalityLatentFrame":
        """
        Element-wise adds two ListModalityLatentFrame objects.
        This is done by adding each corresponding frame in the lists.
        The structure of both lists must be identical.

        Args:
            other (ListModalityLatentFrame): The list of frames to add.

        Returns:
            ListModalityLatentFrame: A new list of frames with the addition result.
        """
        if not isinstance(other, ListModalityLatentFrame):
            return NotImplemented

        if len(self.frames) != len(other.frames):
            raise ValueError(
                f"Cannot add ListModalityLatentFrame objects: number of frames differs "
                f"({len(self.frames)} vs {len(other.frames)})."
            )

        # Use the __add__ method of ModalityLatentFrame in a list comprehension
        # This delegates all the detailed block-level checks to the child object.
        new_frames = [
            self_frame + other_frame
            for self_frame, other_frame in zip(self.frames, other.frames)
        ]

        return ListModalityLatentFrame(frames=new_frames)

    # Add this method inside the ListModalityLatentFrame class
    def calc_loss(
        self,
        modality_loss_func_dict: Dict[str, callable],
        origin_frame: "ListModalityFrame",
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the aggregated loss for each modality across all frames in the list.

        It delegates the calculation for each frame to `ModalityLatentFrame.calc_loss`
        and then aggregates the results.

        Args:
            modality_loss_func_dict (Dict[str, callable]): Dictionary of loss functions.
            origin_frame (ListModalityFrame): The original list of frames with target data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping each modality to its final
                                    aggregated (mean) scalar loss tensor.
        """
        if not isinstance(origin_frame, ListModalityFrame):
            raise TypeError(
                f"origin_frame must be of type ListModalityFrame, but got {type(origin_frame)}"
            )

        if len(self.frames) != len(origin_frame.frames):
            raise ValueError(
                "List structure mismatch: number of frames differs "
                f"({len(self.frames)} vs {len(origin_frame.frames)})."
            )

        # Use defaultdict to aggregate losses across all frames
        total_losses = defaultdict(list)

        # Delegate loss calculation to each frame and aggregate the results
        for pred_frame, target_frame in zip(self.frames, origin_frame.frames):
            frame_losses = pred_frame.calc_loss(modality_loss_func_dict, target_frame)
            for modal, loss_val in frame_losses.items():
                total_losses[modal].append(loss_val)

        # Average the losses for each modality across all frames
        final_losses = {
            modal: torch.mean(torch.stack(loss_list))
            for modal, loss_list in total_losses.items()
        }

        return final_losses


def get_shift_states(
    frame_or_list: Union[ModalityLatentFrame, ListModalityLatentFrame],
) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    提取最后一个时间步作为移位状态。
    - 如果输入是 ModalityLatentFrame, 返回每个块的最后一个时间步的列表。
    - 如果输入是 ListModalityLatentFrame, 返回每种模态的最后一个时间步的字典。
    """
    if isinstance(frame_or_list, ModalityLatentFrame):
        states = []
        for i, block in enumerate(frame_or_list.blocks):
            if len(block) == 0:
                raise ValueError(f"Empty block at index {i} ('{block.modal_type}')")
            # CORRECTED: Keep the channel dimension
            states.append(block.units[:, -1, :])
        return states

    elif isinstance(frame_or_list, ListModalityLatentFrame):
        states_dict = {}
        all_modal_types = sorted(
            list(set(b.modal_type for f in frame_or_list.frames for b in f.blocks))
        )

        for modal_type in all_modal_types:
            last_found_state = None
            # 从后向前遍历以快速找到最后一个非空块
            for frame in reversed(frame_or_list.frames):
                for block in reversed(frame.blocks):
                    if block.modal_type == modal_type and len(block) > 0:
                        last_found_state = block.units[:, -1, :]
                        break
                if last_found_state is not None:
                    break

            if last_found_state is None:
                raise ValueError(
                    f"Cannot get shift state for modal '{modal_type}'; all its blocks are empty."
                )
            states_dict[modal_type] = last_found_state
        return states_dict

    else:
        raise TypeError(
            "Input must be of type ModalityLatentFrame or ListModalityLatentFrame"
        )
