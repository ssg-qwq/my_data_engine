import torch
import torch.nn as nn
import random
from typing import List, Dict, Any, Tuple, Optional, Union, cast, Callable
import heapq
from collections import defaultdict
from .funcs import interleave_lists
from .special_tokens import special_token_config  # 不再需要


class ModalityBlock:
    def __init__(
        self,
        modal_type: str,  # 模态类型，例如"text", "audio", "image"
        units: torch.Tensor,  # 可能为tokens(N,) 或潜空间(N, C)
        masks: Optional[torch.Tensor] = None,  # 用来计算loss的mask(N,)
        attention_masks: Optional[torch.Tensor] = None,  # 用来遮罩注意力的mask(N,)
        mixing_mode: str = "condition",  # 混合模式
        condition_level: int = 0,  # 如果为condition混合模式，该块的前后位置
    ):
        assert mixing_mode in ["condition", "dynamic"]
        self.modal_type = modal_type
        self.units = units
        self.mixing_mode = mixing_mode
        self.condition_level = condition_level

        # 确保units不为空
        if units.shape[0] == 0:
            raise ValueError("ModalityBlock units cannot be empty.")

        N = units.shape[0]
        device = units.device

        if masks is None:
            self.masks = torch.ones((N,), dtype=torch.float32, device=device)
        else:
            self.masks = masks

        if attention_masks is None:
            self.attention_masks = torch.ones((N,), dtype=torch.bfloat16, device=device)
        else:
            self.attention_masks = attention_masks

    def __len__(self):
        return self.units.shape[0]

    def __repr__(self):
        return (
            f"ModalityBlock(modal='{self.modal_type}', mode='{self.mixing_mode}', "
            f"level={self.condition_level}, shape={self.units.shape}')"
        )


class ModalityLatentMixingSequence:
    def __init__(
        self,
        latents: torch.Tensor,  # (T, C) 混合后的模态潜空间
        modals_per_token: List[str],  # 每个T维度上潜空间单元的对应模态
        reconstruction_metadata: Dict[str, Any],  # 用来重构回Blocks的元信息
        masks: Optional[torch.Tensor] = None,  # (T,) 序列对应masks
        attention_masks: Optional[torch.Tensor] = None,  # (T,) 序列对应attention masks
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

        _, C = self.latents.shape
        device = self.latents.device

        reconstructed_blocks = [None] * len(block_specs)

        # --- 1. 重建 Condition Blocks ---
        total_condition_len = sum(block_specs[i]["length"] for i in condition_order)

        if total_condition_len > 0:
            condition_latents = self.latents[:total_condition_len, :]
            condition_masks = self.masks[:total_condition_len]
            condition_att_masks = self.attention_masks[:total_condition_len]

        dynamic_latents = self.latents[total_condition_len:, :]
        dynamic_masks = self.masks[total_condition_len:]
        dynamic_att_masks = self.attention_masks[total_condition_len:]

        current_pos = 0
        for block_idx in condition_order:
            spec = block_specs[block_idx]
            length = spec["length"]

            block_units = condition_latents[current_pos : current_pos + length, :]
            block_masks = condition_masks[current_pos : current_pos + length]
            block_att_masks = condition_att_masks[current_pos : current_pos + length]

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
            # 每个token现在是一个独立的chunk
            placeholder_chunks_by_block = []
            for i, spec in enumerate(dynamic_blocks_to_process):
                # (原始 dynamic 列表中的索引, 在块内的起始位置, 长度)
                block_chunks = [(i, k, 1) for k in range(spec["length"])]
                placeholder_chunks_by_block.append(block_chunks)

            interleaved_placeholder = interleave_lists(*placeholder_chunks_by_block)

            # 初始化用于存放重建后数据的列表
            temp_dynamic_units = [
                torch.empty(
                    (spec["length"], C), device=device, dtype=self.latents.dtype
                )
                for spec in dynamic_blocks_to_process
            ]
            temp_dynamic_masks = [
                torch.empty((spec["length"],), device=device, dtype=self.masks.dtype)
                for spec in dynamic_blocks_to_process
            ]
            temp_dynamic_att_masks = [
                torch.empty(
                    (spec["length"],), device=device, dtype=self.attention_masks.dtype
                )
                for spec in dynamic_blocks_to_process
            ]

            # 根据交错顺序，将数据“分发”回各自的块中
            current_pos = 0
            for block_info in interleaved_placeholder:
                block_list_idx, start_idx, chunk_len = (
                    block_info  # chunk_len is always 1
                )

                unit_chunk = dynamic_latents[current_pos : current_pos + chunk_len, :]
                mask_chunk = dynamic_masks[current_pos : current_pos + chunk_len]
                att_mask_chunk = dynamic_att_masks[
                    current_pos : current_pos + chunk_len
                ]

                temp_dynamic_units[block_list_idx][
                    start_idx : start_idx + chunk_len, :
                ] = unit_chunk
                temp_dynamic_masks[block_list_idx][
                    start_idx : start_idx + chunk_len
                ] = mask_chunk
                temp_dynamic_att_masks[block_list_idx][
                    start_idx : start_idx + chunk_len
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
    def ctx_len(self) -> int:
        if not self.blocks:
            raise ValueError(
                "Cannot determine context length from an empty ModalityFrame."
            )
        return sum(len(block) for block in self.blocks)

    def shuffle(self):
        random.shuffle(self.blocks)

    def _apply_by_key_parallel(
        self,
        module_dict: Dict[str, nn.Module],
        key_func: Callable[[ModalityBlock], str],
    ) -> "ModalityFrame":
        grouped_units = defaultdict(list)
        grouped_metadata = defaultdict(list)
        unprocessed_blocks = []

        for i, block in enumerate(self.blocks):
            if len(block) == 0:
                unprocessed_blocks.append((i, block))
                continue

            key = key_func(block)
            if key in module_dict:
                grouped_units[key].append(block.units)
                grouped_metadata[key].append(
                    {"original_index": i, "length": len(block), "original_block": block}
                )
            else:
                unprocessed_blocks.append((i, block))

        new_blocks = [None] * len(self.blocks)

        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=0)
            module = module_dict[key]
            processed_units = module(concatenated_units)

            if processed_units.shape[0] != concatenated_units.shape[0] or (
                processed_units.ndim > 1
                and concatenated_units.ndim > 1
                and processed_units.shape[1:] != concatenated_units.shape[1:]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed dimensions! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            metadata_list = grouped_metadata[key]
            lengths = [meta["length"] for meta in metadata_list]
            split_processed_units = torch.split(processed_units, lengths, dim=0)

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

        for index, block in unprocessed_blocks:
            new_blocks[index] = block

        if any(b is None for b in new_blocks):
            raise RuntimeError("Failed to reconstruct all blocks in ModalityFrame.")

        return ModalityFrame(blocks=new_blocks)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityFrame":
        return self._apply_by_key_parallel(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityFrame":
        return self._apply_by_key_parallel(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    def apply_module_parallel(self, module: nn.Module) -> "ModalityFrame":
        if not self.blocks:
            raise ValueError("Cannot apply module to an empty ModalityFrame.")

        all_units = []
        lengths = []
        original_blocks_with_units = []
        for block in self.blocks:
            if len(block) > 0:
                all_units.append(block.units)
                lengths.append(len(block))
                original_blocks_with_units.append(block)

        if not all_units:
            return ModalityFrame(blocks=[b for b in self.blocks])

        concatenated_units = torch.cat(all_units, dim=0)
        processed_units = module(concatenated_units)

        if processed_units.shape[0] != concatenated_units.shape[0]:
            raise ValueError(
                f"The provided module changed the sequence length! "
                f"Input length: {concatenated_units.shape[0]}, "
                f"Output length: {processed_units.shape[0]}"
            )

        split_processed_units = torch.split(processed_units, lengths, dim=0)

        new_blocks = []
        processed_idx = 0
        for original_block in self.blocks:
            if len(original_block) > 0:
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=split_processed_units[processed_idx],
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                processed_idx += 1
            else:  # Handle empty blocks
                new_block = original_block
            new_blocks.append(new_block)

        return ModalityFrame(blocks=new_blocks)

    def embed(
        self, embedding_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        processed_frame = self.apply_by_modal(embedding_dict)
        return ModalityLatentFrame(blocks=processed_frame.blocks)

    def calc_loss(
        self,
        modality_loss_func_dict: Dict[str, callable],
        origin_frame: "ModalityFrame",
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(origin_frame, ModalityFrame):
            raise TypeError(
                f"origin_frame must be of type ModalityFrame, but got {type(origin_frame)}"
            )
        if len(self.blocks) != len(origin_frame.blocks):
            raise ValueError(
                f"Frame structure mismatch: number of blocks differs ({len(self.blocks)} vs {len(origin_frame.blocks)})."
            )

        aggregated_losses = defaultdict(list)
        for pred_block, target_block in zip(self.blocks, origin_frame.blocks):
            modal_type = pred_block.modal_type
            if modal_type != target_block.modal_type:
                raise ValueError(
                    f"Frame structure mismatch: modal types at the same position differ."
                )

            if modal_type in modality_loss_func_dict:
                loss_func = modality_loss_func_dict[modal_type]
                loss_mask = pred_block.masks
                bool_mask = loss_mask.bool()
                if not bool_mask.any():
                    continue

                masked_preds = pred_block.units[bool_mask]
                masked_targets = target_block.units[bool_mask]
                loss = loss_func(masked_preds, masked_targets)
                aggregated_losses[modal_type].append(loss)

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
        grouped_units = defaultdict(list)
        grouped_metadata = defaultdict(list)
        unprocessed_blocks_with_indices = []

        for i, block in enumerate(self.blocks):
            if len(block) == 0:
                unprocessed_blocks_with_indices.append((i, block))
                continue
            key = key_func(block)
            if key in module_dict:
                grouped_units[key].append(block.units)
                grouped_metadata[key].append(
                    {"original_index": i, "length": len(block), "original_block": block}
                )
            else:
                unprocessed_blocks_with_indices.append((i, block))

        new_blocks = [None] * len(self.blocks)

        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=0)
            module = module_dict[key]
            processed_units = module(concatenated_units)

            if processed_units.shape[0] != concatenated_units.shape[0] or (
                processed_units.ndim > 1
                and concatenated_units.ndim > 1
                and processed_units.shape[1:] != concatenated_units.shape[1:]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed dimensions! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            metadata_list = grouped_metadata[key]
            lengths = [meta["length"] for meta in metadata_list]
            split_processed_units = torch.split(processed_units, lengths, dim=0)

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

        for index, block in unprocessed_blocks_with_indices:
            new_blocks[index] = block

        if any(b is None for b in new_blocks):
            raise RuntimeError(
                "Failed to reconstruct all blocks in ModalityLatentFrame."
            )

        return ModalityLatentFrame(blocks=new_blocks)

    def arrange_frame(self) -> "ModalityLatentMixingSequence":
        if not self.blocks:
            raise ValueError("ModalityLatentFrame is empty. Cannot arrange.")

        valid_blocks = [block for block in self.blocks if len(block) > 0]
        if not valid_blocks:
            raise ValueError(
                "All blocks in ModalityLatentFrame are empty. Cannot arrange."
            )

        ref_block = valid_blocks[0]
        _, C = ref_block.units.shape
        device = ref_block.units.device

        # --- 元数据准备 ---
        block_specs = []
        for i, block in enumerate(self.blocks):
            spec = {
                "original_index": i,
                "modal_type": block.modal_type,
                "mixing_mode": block.mixing_mode,
                "condition_level": block.condition_level,
                "length": len(block),
            }
            block_specs.append(spec)

        # --- 1. 分离和排序 Condition Blocks ---
        condition_blocks_with_indices = [
            (i, block)
            for i, block in enumerate(self.blocks)
            if block.mixing_mode == "condition" and len(block) > 0
        ]
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
            modals_per_token_condition.extend([block.modal_type] * len(block))
            sorted_condition_indices.append(original_index)

        # --- 2. 准备和交错 Dynamic Blocks ---
        dynamic_blocks_with_indices = [
            (i, block)
            for i, block in enumerate(self.blocks)
            if block.mixing_mode == "dynamic" and len(block) > 0
        ]

        dynamic_chunks_by_block = []
        original_dynamic_indices = [i for i, b in dynamic_blocks_with_indices]

        for _, block in dynamic_blocks_with_indices:
            chunks = []
            for i in range(len(block)):
                # 每个token现在是一个独立的chunk
                chunks.append(
                    (
                        block.units[i : i + 1, :],
                        block.masks[i : i + 1],
                        block.attention_masks[i : i + 1],
                        block.modal_type,
                    )
                )
            dynamic_chunks_by_block.append(chunks)

        interleaved_chunks = interleave_lists(*dynamic_chunks_by_block)

        final_dynamic_latents = [chunk[0] for chunk in interleaved_chunks]
        final_dynamic_masks = [chunk[1] for chunk in interleaved_chunks]
        final_dynamic_att_masks = [chunk[2] for chunk in interleaved_chunks]
        modals_per_token_dynamic = [chunk[3] for chunk in interleaved_chunks]

        # --- 3. 合并所有部分 ---
        all_latents = final_condition_latents + final_dynamic_latents
        all_masks = final_condition_masks + final_dynamic_masks
        all_att_masks = final_condition_att_masks + final_dynamic_att_masks

        final_latents = (
            torch.cat(all_latents, dim=0)
            if all_latents
            else torch.empty(0, C, device=device)
        )
        final_masks = (
            torch.cat(all_masks, dim=0) if all_masks else torch.empty(0, device=device)
        )
        final_att_masks = (
            torch.cat(all_att_masks, dim=0)
            if all_att_masks
            else torch.empty(0, device=device)
        )

        final_modals_per_token = modals_per_token_condition + modals_per_token_dynamic

        reconstruction_metadata = {
            "block_specs": block_specs,
            "condition_order": sorted_condition_indices,
            "dynamic_indices": original_dynamic_indices,
        }

        return ModalityLatentMixingSequence(
            latents=final_latents,
            masks=final_masks,
            attention_masks=final_att_masks,
            modals_per_token=final_modals_per_token,
            reconstruction_metadata=reconstruction_metadata,
        )

    def shift(
        self, shift_states_dict: Dict[str, torch.Tensor]
    ) -> "ModalityLatentFrame":
        if not self.blocks:
            return ModalityLatentFrame(blocks=[])

        all_units = []
        all_prev_units_parts = []
        split_lengths = []

        _, C = self.blocks[0].units.shape

        for block in self.blocks:
            x = block.units
            if x.shape[1] != C:
                raise ValueError(
                    "All blocks must have the same channel dimensions for optimized shift."
                )

            kk = f"{block.modal_type}_{block.mixing_mode}"
            shift_state = shift_states_dict[kk]

            if shift_state.shape != (C,):
                raise ValueError(
                    f"Invalid shape for shift_state for modal '{kk}'. Expected ({C},), but got {shift_state.shape}."
                )

            all_units.append(x)
            split_lengths.append(x.shape[0])

            shift_expanded = shift_state.unsqueeze(0)  # (1, C)
            x_t_minus_1 = x[:-1]  # (N-1, C)
            prev_unit_part = torch.cat([shift_expanded, x_t_minus_1], dim=0)
            all_prev_units_parts.append(prev_unit_part)

        x_concat = torch.cat(all_units, dim=0)
        x_prev_concat = torch.cat(all_prev_units_parts, dim=0)
        xx_concat = x_prev_concat - x_concat
        split_results = torch.split(xx_concat, split_lengths, dim=0)

        new_blocks = []
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
        if not self.blocks:
            raise ValueError("Cannot apply module to an empty ModalityLatentFrame.")

        all_units, lengths, original_blocks_with_units = [], [], []
        for block in self.blocks:
            if len(block) > 0:
                all_units.append(block.units)
                lengths.append(len(block))
                original_blocks_with_units.append(block)

        if not all_units:
            return ModalityLatentFrame(blocks=[b for b in self.blocks])

        concatenated_units = torch.cat(all_units, dim=0)
        processed_units = module(concatenated_units)

        if processed_units.shape[0] != concatenated_units.shape[0]:
            raise ValueError(
                f"The provided module changed the sequence length! Input: {concatenated_units.shape[0]}, Output: {processed_units.shape[0]}"
            )

        split_processed_units = torch.split(processed_units, lengths, dim=0)

        new_blocks = []
        processed_idx = 0
        for original_block in self.blocks:
            if len(original_block) > 0:
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=split_processed_units[processed_idx],
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                processed_idx += 1
            else:
                new_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=original_block.units.clone(),
                    masks=original_block.masks.clone(),
                    attention_masks=original_block.attention_masks.clone(),
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
            new_blocks.append(new_block)

        return ModalityLatentFrame(blocks=new_blocks)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        return self._apply_by_key_parallel(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ModalityLatentFrame":
        return self._apply_by_key_parallel(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    @property
    def ctx_len(self) -> int:
        if not self.blocks:
            raise ValueError(
                "Cannot determine context length from an empty ModalityLatentFrame."
            )
        return sum(len(block) for block in self.blocks)

    def __add__(self, other: "ModalityLatentFrame") -> "ModalityLatentFrame":
        """
        以并行方式将两个 ModalityLatentFrame 的 latent space 相加。
        该方法通过拼接所有块的张量，执行一次批处理加法，然后拆分回原结构，以提高计算效率。
        """
        if not isinstance(other, ModalityLatentFrame):
            return NotImplemented

        if len(self.blocks) != len(other.blocks):
            raise ValueError(
                f"Cannot add ModalityLatentFrames: number of blocks differs ({len(self.blocks)} vs {len(other.blocks)})."
            )

        # 1. 收集和验证
        self_units_list = []
        other_units_list = []
        lengths = []
        metadata_blocks = []  # 存储有内容的原始块以供重构
        empty_block_indices = []  # 存储空块的索引

        for i, (self_block, other_block) in enumerate(zip(self.blocks, other.blocks)):
            # 预验证，确保结构一致
            if self_block.modal_type != other_block.modal_type:
                raise ValueError(
                    f"Cannot add blocks at index {i}: modal types do not match ('{self_block.modal_type}' vs '{other_block.modal_type}')."
                )
            if self_block.units.shape != other_block.units.shape:
                raise ValueError(
                    f"Cannot add blocks for modal '{self_block.modal_type}' at index {i}: shapes are incompatible ({self_block.units.shape} vs {other_block.units.shape})."
                )

            if len(self_block) > 0:
                self_units_list.append(self_block.units)
                other_units_list.append(other_block.units)
                lengths.append(len(self_block))
                metadata_blocks.append(self_block)
            else:
                empty_block_indices.append(i)

        # 如果所有块都为空，直接返回一个相同结构的空Frame
        if not self_units_list:
            return ModalityLatentFrame(blocks=[b for b in self.blocks])

        # 2. 批量计算
        self_concat = torch.cat(self_units_list, dim=0)
        other_concat = torch.cat(other_units_list, dim=0)
        result_concat = self_concat + other_concat

        # 3. 拆分
        split_results = torch.split(result_concat, lengths, dim=0)

        # 4. 重构
        new_blocks = [None] * len(self.blocks)

        # 填充处理过的块
        for i, original_block in enumerate(metadata_blocks):
            new_units = split_results[i]
            result_block = ModalityBlock(
                modal_type=original_block.modal_type,
                units=new_units,
                masks=original_block.masks,
                attention_masks=original_block.attention_masks,
                mixing_mode=original_block.mixing_mode,
                condition_level=original_block.condition_level,
            )
            # 找到它在原始列表中的位置
            # 这个方法有点低效，更好的方法是在第一步就记录原始索引
            original_index = self.blocks.index(original_block)
            new_blocks[original_index] = result_block

        # 填充空块
        for i in empty_block_indices:
            new_blocks[i] = self.blocks[i]

        # 改进的重构逻辑
        new_blocks_reordered = []
        processed_idx = 0
        for i, original_block in enumerate(self.blocks):
            if len(original_block) > 0:
                new_units = split_results[processed_idx]
                result_block = ModalityBlock(
                    modal_type=original_block.modal_type,
                    units=new_units,
                    masks=original_block.masks,
                    attention_masks=original_block.attention_masks,
                    mixing_mode=original_block.mixing_mode,
                    condition_level=original_block.condition_level,
                )
                new_blocks_reordered.append(result_block)
                processed_idx += 1
            else:
                # 对于空块，直接添加
                new_blocks_reordered.append(original_block)

        return ModalityLatentFrame(blocks=new_blocks_reordered)


class ListModalityLatentMixingSequence(ModalityLatentMixingSequence):
    def __init__(
        self,
        latents: torch.Tensor,
        modals_per_token: List[str],
        reconstruction_metadata: Dict[str, Any],
        masks: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            latents=latents,
            modals_per_token=modals_per_token,
            reconstruction_metadata=reconstruction_metadata,
            masks=masks,
            attention_masks=attention_masks,
        )

    def rearrange_frame(self) -> "ListModalityLatentFrame":
        """
        该函数用来将 ListModalityLatentFrame.arrange_frame 得到的本对象
        重构回原 ListModalityLatentFrame。
        """
        # 1. 从元数据中解包信息
        frame_lengths = self.reconstruction_metadata["frame_lengths"]
        individual_metadata = self.reconstruction_metadata["individual_metadata"]

        # 如果原始列表为空，则直接返回一个空的 ListModalityLatentFrame
        if not frame_lengths:
            return ListModalityLatentFrame(frames=[])

        # 2. 使用 frame_lengths 将大张量切分回片段列表
        split_latents = list(torch.split(self.latents, frame_lengths, dim=0))
        split_masks = list(torch.split(self.masks, frame_lengths, dim=0))
        split_att_masks = list(torch.split(self.attention_masks, frame_lengths, dim=0))

        # 3. 手动切分 modals_per_token 列表
        split_modals = []
        current_pos = 0
        for length in frame_lengths:
            split_modals.append(self.modals_per_token[current_pos : current_pos + length])
            current_pos += length
        
        rearranged_frames = []
        # 4. 遍历所有片段，并逐个重构
        for i in range(len(frame_lengths)):
            # 5. 为每个原始 frame 创建一个临时的 ModalityLatentMixingSequence
            temp_sequence = ModalityLatentMixingSequence(
                latents=split_latents[i],
                modals_per_token=split_modals[i],
                reconstruction_metadata=individual_metadata[i],
                masks=split_masks[i],
                attention_masks=split_att_masks[i]
            )

            # 6. 调用基类的 rearrange_frame 方法来重构单个 Frame
            rearranged_frame = temp_sequence.rearrange_frame()
            rearranged_frames.append(rearranged_frame)

        # 7. & 8. 用重构好的 Frame 列表创建并返回最终结果
        return ListModalityLatentFrame(frames=rearranged_frames)


class ListModalityFrame:
    def __init__(self, frames: List[ModalityFrame]):
        self.frames = frames

    @property
    def ctx_len(self) -> int:
        if not self.frames:
            return 0
        return sum(len(block) for frame in self.frames for block in frame.blocks)

    def shuffle_frame(self):
        for frame in self.frames:
            frame.shuffle()

    def _apply_by_key_parallel_list(
        self,
        module_dict: Dict[str, nn.Module],
        key_func: Callable[[ModalityBlock], str],
    ) -> "ListModalityFrame":
        if not self.frames:
            return ListModalityFrame(frames=[])

        grouped_units = defaultdict(list)
        block_positions = []
        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                key = key_func(block)
                position_info = {
                    "frame_idx": frame_idx,
                    "block_idx": block_idx,
                    "processed": False,
                    "block": block,
                    "key": key,
                }
                if len(block) > 0 and key in module_dict:
                    grouped_units[key].append(block.units)
                    position_info["processed"] = True
                block_positions.append(position_info)

        processed_unit_chunks = defaultdict(list)
        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=0)
            module = module_dict[key]
            processed_units = module(concatenated_units)

            if processed_units.shape[0] != concatenated_units.shape[0] or (
                processed_units.ndim > 1
                and concatenated_units.ndim > 1
                and processed_units.shape[1:] != concatenated_units.shape[1:]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed dimensions! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            lengths = [
                len(pos["block"])
                for pos in block_positions
                if pos["processed"] and pos["key"] == key
            ]
            processed_unit_chunks[key] = list(
                torch.split(processed_units, lengths, dim=0)
            )

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

        new_frames = []
        for frame_idx in range(len(self.frames)):
            sorted_blocks = sorted(reconstruction_data[frame_idx], key=lambda x: x[0])
            new_blocks_for_frame = [b for _, b in sorted_blocks]
            new_frames.append(ModalityFrame(blocks=new_blocks_for_frame))

        return ListModalityFrame(frames=new_frames)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityFrame":
        return self._apply_by_key_parallel_list(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityFrame":
        return self._apply_by_key_parallel_list(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    def embed(
        self, embedding_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        processed_list_frame = self.apply_by_modal(embedding_dict)
        latent_frames = [
            ModalityLatentFrame(blocks=frame.blocks)
            for frame in processed_list_frame.frames
        ]
        return ListModalityLatentFrame(frames=latent_frames)

    def apply_module_parallel(self, module: nn.Module) -> "ListModalityFrame":
        if not self.frames:
            return self

        all_units, metadata = [], []
        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                if len(block) > 0:
                    all_units.append(block.units)
                    metadata.append(
                        {
                            "frame_idx": frame_idx,
                            "block_idx": block_idx,
                            "length": len(block),
                            "original_block": block,
                        }
                    )

        if not all_units:
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

        concatenated_units = torch.cat(all_units, dim=0)
        processed_units = module(concatenated_units)

        if processed_units.shape[0] != concatenated_units.shape[0]:
            raise ValueError(
                f"Module changed sequence length! Input: {concatenated_units.shape[0]}, Output: {processed_units.shape[0]}"
            )

        lengths = [meta["length"] for meta in metadata]
        split_processed_units = torch.split(processed_units, lengths, dim=0)

        reconstruction_data = [[] for _ in self.frames]
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

        new_frames = []
        for frame_idx, frame in enumerate(self.frames):
            new_blocks_for_frame = []
            processed_blocks_map = {
                block_idx: block for block_idx, block in reconstruction_data[frame_idx]
            }
            for block_idx, original_block in enumerate(frame.blocks):
                if block_idx in processed_blocks_map:
                    new_blocks_for_frame.append(processed_blocks_map[block_idx])
                else:
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

    def shift(
        self, shift_states_dict: Dict[str, torch.Tensor]
    ) -> "ListModalityLatentFrame":
        if not self.frames:
            return ListModalityLatentFrame(frames=[])

        new_frames = []
        current_shift_states = shift_states_dict.copy()
        for frame in self.frames:
            shifted_frame = frame.shift(current_shift_states)
            new_frames.append(shifted_frame)

            next_states = {}
            for block in frame.blocks:
                if len(block) > 0:
                    kk = f"{block.modal_type}_{block.mixing_mode}"
                    last_timestep = block.units[-1, :]
                    next_states[kk] = last_timestep
            current_shift_states = next_states

        return ListModalityLatentFrame(frames=new_frames)


    def arrange_frame(self) -> "ListModalityLatentMixingSequence":
        """
        将一个 List[ModalityLatentFrame] 展平为一个单一的、连续的序列。
        它首先对列表中的每个 Frame 单独进行 arrange，然后再将结果拼接起来。
        """
        if not self.frames:
            # 如果列表为空，则无法确定设备、数据类型和通道数 C。
            # 在实践中，应避免处理完全空的列表。这里抛出异常以保证安全。
            raise ValueError("Cannot arrange an empty list of frames.")

        # 1. 对列表中的每个 frame 单独调用 arrange_frame
        arranged_sequences = [frame.arrange_frame() for frame in self.frames]

        # 如果所有 frame 都是空的，也需要处理
        if not any(len(seq.latents) > 0 for seq in arranged_sequences):
             raise ValueError("All frames in the list are empty, cannot arrange.")
                
        # 2. 收集所有拼接所需的部分
        all_latents = [seq.latents for seq in arranged_sequences if len(seq.latents) > 0]
        all_masks = [seq.masks for seq in arranged_sequences if len(seq.masks) > 0]
        all_att_masks = [seq.attention_masks for seq in arranged_sequences if len(seq.attention_masks) > 0]
        # 使用 sum(list_of_lists, []) 来高效地展平一个二维列表
        all_modals_per_token = sum([seq.modals_per_token for seq in arranged_sequences], [])

        # 3. 沿序列维度（dim=0）拼接
        final_latents = torch.cat(all_latents, dim=0)
        final_masks = torch.cat(all_masks, dim=0)
        final_att_masks = torch.cat(all_att_masks, dim=0)

        # 4. 创建用于重构 List[ModalityLatentFrame] 的元数据
        # 这个元数据包含了恢复原始列表结构所需的信息
        list_reconstruction_metadata = {
            # 存储每个 frame 展平后的序列长度
            "frame_lengths": [len(seq.latents) for seq in arranged_sequences],
            # 存储每个 frame 自身的重构元数据
            "individual_metadata": [seq.reconstruction_metadata for seq in arranged_sequences]
        }

        # 5. 返回最终的、单一的、连续的序列对象
        return ListModalityLatentMixingSequence(
            latents=final_latents,
            masks=final_masks,
            attention_masks=final_att_masks,
            modals_per_token=all_modals_per_token,
            reconstruction_metadata=list_reconstruction_metadata
        )

    def apply_module_parallel(self, module: nn.Module) -> "ListModalityLatentFrame":
        if not self.frames:
            return self

        all_units, metadata = [], []
        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                if len(block) > 0:
                    all_units.append(block.units)
                    metadata.append(
                        {
                            "frame_idx": frame_idx,
                            "block_idx": block_idx,
                            "length": len(block),
                            "original_block": block,
                        }
                    )

        if not all_units:
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

        concatenated_units = torch.cat(all_units, dim=0)
        processed_units = module(concatenated_units)

        if processed_units.shape[0] != concatenated_units.shape[0]:
            raise ValueError(
                f"Module changed sequence length! Input: {concatenated_units.shape[0]}, Output: {processed_units.shape[0]}"
            )

        lengths = [meta["length"] for meta in metadata]
        split_processed_units = torch.split(processed_units, lengths, dim=0)

        reconstruction_data = [[] for _ in self.frames]
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

        new_frames = []
        for frame_idx, frame in enumerate(self.frames):
            new_blocks_for_frame = []
            processed_blocks_map = {
                block_idx: block for block_idx, block in reconstruction_data[frame_idx]
            }
            for block_idx, original_block in enumerate(frame.blocks):
                if block_idx in processed_blocks_map:
                    new_blocks_for_frame.append(processed_blocks_map[block_idx])
                else:
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
        self,
        module_dict: Dict[str, nn.Module],
        key_func: Callable[[ModalityBlock], str],
    ) -> "ListModalityLatentFrame":
        if not self.frames:
            return ListModalityLatentFrame(frames=[])

        grouped_units = defaultdict(list)
        block_positions = []
        for frame_idx, frame in enumerate(self.frames):
            for block_idx, block in enumerate(frame.blocks):
                key = key_func(block)
                position_info = {
                    "frame_idx": frame_idx,
                    "block_idx": block_idx,
                    "processed": False,
                    "block": block,
                    "key": key,
                }
                if len(block) > 0 and key in module_dict:
                    grouped_units[key].append(block.units)
                    position_info["processed"] = True
                block_positions.append(position_info)

        processed_unit_chunks = defaultdict(list)
        for key, units_list in grouped_units.items():
            concatenated_units = torch.cat(units_list, dim=0)
            processed_units = module_dict[key](concatenated_units)

            if processed_units.shape[0] != concatenated_units.shape[0] or (
                processed_units.ndim > 1
                and concatenated_units.ndim > 1
                and processed_units.shape[1:] != concatenated_units.shape[1:]
            ):
                raise ValueError(
                    f"Module for key '{key}' changed dimensions! "
                    f"Input: {concatenated_units.shape}, Output: {processed_units.shape}"
                )

            lengths = [
                len(pos["block"])
                for pos in block_positions
                if pos["processed"] and pos["key"] == key
            ]
            processed_unit_chunks[key] = list(
                torch.split(processed_units, lengths, dim=0)
            )

        reconstruction_data = [[] for _ in self.frames]
        counters = defaultdict(int)
        for pos_info in block_positions:
            frame_idx, block_idx = pos_info["frame_idx"], pos_info["block_idx"]
            original_block = pos_info["block"]

            if pos_info["processed"]:
                key = pos_info["key"]
                new_unit_chunk = processed_unit_chunks[key][counters[key]]
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

        new_frames = []
        for frame_idx in range(len(self.frames)):
            sorted_blocks = sorted(reconstruction_data[frame_idx], key=lambda x: x[0])
            new_blocks = [b for _, b in sorted_blocks]
            new_frames.append(ModalityLatentFrame(blocks=new_blocks))

        return ListModalityLatentFrame(frames=new_frames)

    def apply_by_modal(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        return self._apply_by_key_parallel_list(module_dict, lambda b: b.modal_type)

    def apply_by_modal_and_mixing_mode(
        self, module_dict: Dict[str, torch.nn.Module]
    ) -> "ListModalityLatentFrame":
        return self._apply_by_key_parallel_list(
            module_dict, lambda b: f"{b.modal_type}_{b.mixing_mode}"
        )

    @property
    def ctx_len(self) -> int:
        if not self.frames:
            return 0
        return sum(len(block) for frame in self.frames for block in frame.blocks)

    def __add__(self, other: "ListModalityLatentFrame") -> "ListModalityLatentFrame":
        """
        以并行方式将两个 ListModalityLatentFrame 列表的 latent space 相加。
        该方法将所有Frames中的所有Blocks的张量展平、拼接，执行一次批处理加法，然后重构回原始的列表结构。
        """
        if not isinstance(other, ListModalityLatentFrame):
            return NotImplemented

        if len(self.frames) != len(other.frames):
            raise ValueError(
                f"Cannot add ListModalityLatentFrame objects: number of frames differs ({len(self.frames)} vs {len(other.frames)})."
            )

        # 1. 收集和验证
        self_all_units = []
        other_all_units = []
        metadata = []  # 存储重构所需信息

        for frame_idx, (self_frame, other_frame) in enumerate(
            zip(self.frames, other.frames)
        ):
            if len(self_frame.blocks) != len(other_frame.blocks):
                raise ValueError(
                    f"Cannot add frames at index {frame_idx}: number of blocks differs ({len(self_frame.blocks)} vs {len(other_frame.blocks)})."
                )

            for block_idx, (self_block, other_block) in enumerate(
                zip(self_frame.blocks, other_frame.blocks)
            ):
                # 预验证
                if self_block.modal_type != other_block.modal_type:
                    raise ValueError(
                        f"Cannot add blocks at frame {frame_idx}, block {block_idx}: modal types do not match ('{self_block.modal_type}' vs '{other_block.modal_type}')."
                    )
                if self_block.units.shape != other_block.units.shape:
                    raise ValueError(
                        f"Cannot add blocks at frame {frame_idx}, block {block_idx}: shapes are incompatible ({self_block.units.shape} vs {other_block.units.shape})."
                    )

                if len(self_block) > 0:
                    self_all_units.append(self_block.units)
                    other_all_units.append(other_block.units)
                    metadata.append(
                        {
                            "frame_idx": frame_idx,
                            "block_idx": block_idx,
                            "length": len(self_block),
                            "original_block": self_block,
                        }
                    )

        # 如果所有块都为空，直接返回一个相同结构的空ListFrame
        if not self_all_units:
            return ListModalityLatentFrame(
                frames=[
                    ModalityLatentFrame(blocks=[b for b in f.blocks])
                    for f in self.frames
                ]
            )

        # 2. 批量计算
        self_concat = torch.cat(self_all_units, dim=0)
        other_concat = torch.cat(other_all_units, dim=0)
        result_concat = self_concat + other_concat

        # 3. 拆分
        lengths = [meta["length"] for meta in metadata]
        split_results = torch.split(result_concat, lengths, dim=0)

        # 4. 重构
        reconstruction_data = [[] for _ in self.frames]
        for i, meta_info in enumerate(metadata):
            original_block = meta_info["original_block"]
            new_block = ModalityBlock(
                modal_type=original_block.modal_type,
                units=split_results[i],
                masks=original_block.masks,
                attention_masks=original_block.attention_masks,
                mixing_mode=original_block.mixing_mode,
                condition_level=original_block.condition_level,
            )
            reconstruction_data[meta_info["frame_idx"]].append(
                (meta_info["block_idx"], new_block)
            )

        new_frames = []
        for frame_idx, frame in enumerate(self.frames):
            new_blocks_for_frame = []
            processed_blocks_map = {
                block_idx: block for block_idx, block in reconstruction_data[frame_idx]
            }
            for block_idx, original_block in enumerate(frame.blocks):
                if block_idx in processed_blocks_map:
                    new_blocks_for_frame.append(processed_blocks_map[block_idx])
                else:  # 处理空块
                    new_blocks_for_frame.append(original_block)
            new_frames.append(ModalityLatentFrame(blocks=new_blocks_for_frame))

        return ListModalityLatentFrame(frames=new_frames)

    def calc_loss(
        self,
        modality_loss_func_dict: Dict[str, callable],
        origin_frame: "ListModalityFrame",
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(origin_frame, ListModalityFrame):
            raise TypeError(
                f"origin_frame must be of type ListModalityFrame, but got {type(origin_frame)}"
            )
        if len(self.frames) != len(origin_frame.frames):
            raise ValueError(
                f"List structure mismatch: number of frames differs ({len(self.frames)} vs {len(origin_frame.frames)})."
            )

        total_losses = defaultdict(list)
        for pred_frame, target_frame in zip(self.frames, origin_frame.frames):
            frame_losses = pred_frame.calc_loss(modality_loss_func_dict, target_frame)
            for modal, loss_val in frame_losses.items():
                total_losses[modal].append(loss_val)

        final_losses = {
            modal: torch.mean(torch.stack(loss_list))
            for modal, loss_list in total_losses.items()
        }
        return final_losses


def update_shift_states(
    states_dict,
    frame_or_list: Union[ModalityLatentFrame, ListModalityLatentFrame],
) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
    if isinstance(frame_or_list, ModalityLatentFrame):
        for i, block in enumerate(frame_or_list.blocks):
            if len(block) == 0:
                raise ValueError(f"Empty block at index {i} ('{block.modal_type}')")
            states_dict[f"{block.modal_type}_{block.mixing_mode}"] = block.units[-1, :]
        return states_dict

    elif isinstance(frame_or_list, ListModalityLatentFrame):
        for i, block in enumerate(frame_or_list.frames[-1].blocks):
            if len(block) == 0:
                raise ValueError(f"Empty block at index {i} ('{block.modal_type}')")
            states_dict[f"{block.modal_type}_{block.mixing_mode}"] = block.units[-1, :]
        return states_dict

    else:
        raise TypeError(
            "Input must be of type ModalityLatentFrame or ListModalityLatentFrame"
        )
