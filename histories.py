from .frames import ModalityFrame, ListModalityFrame
from typing import List, Tuple

"""
“假帧”是指把一次会话（Conversation）包装成为一个帧（ModalityLatentFrame），
以使多模态模型可以训练串行的预训练数据。
Turns[Rollouts[ListModalityLatentFrame]]
`ModalityLatentFrame` 代表一位角色的一轮发言
`ListModalityLatentFrame` 是一轮对话，大部分时候其长度为2（AI使用了工具可能不为2）
`Rollouts` 记录一轮对话内的重新生成，如果是sft数据集的话，其len为1
`Turns` 记录多轮对话
"""

"""
“统合帧”是所有模态的数据在时间轴上铺开的结果。
Turns[Rollouts[ListModalityLatentFrame]]
其最小的单位为 `ModalityLatentFrame` ，包含了帧率最低的模态信息的1帧，以及其他模态信息在此期间的所有帧。
`ModalityLatentFrame` 被ListModalityLatentFrame包装。`ListModalityLatentFrame` 代表作为一次Rollout的历史长度
`Rollouts` 记录一轮对话内的重新生成，如果不是模拟空间的重新生成，其len为1
`Turns` 在时间轴上记录多个 `Rollouts`
"""

class RolloutHistory:
    def __init__(
        self,
        turns: List[List[Tuple[ListModalityFrame, float]]] = [],
        untrained_history_pointer=None,
    ):
        """
        Turns[Rollouts[(ListModalityFrame, score)]]
        """
        self.turns = turns
        self.untrained_history_pointer = (
            untrained_history_pointer if untrained_history_pointer is not None else len(self.turns)
        )
        
        self.now_state_dir=None
        self.last_state_dir=None
        self.state_after_train_dir=None

    @property
    def chosen_history_ctx_len(self):
        hist=self.get_chosen_history()
        ctx_len=sum(frames.ctx_len for frames in hist)
        return ctx_len
    
    @property
    def untrained_history_ctx_len(self):
        hist=self.get_chosen_history(only_get_untrained=True)
        ctx_len=sum(frames.ctx_len for frames in hist)
        return ctx_len

    def add_turn(self, rollout: Tuple[ListModalityFrame, float]):
        """
        增加一轮消息（无重新生成）
        """
        self.turns.append([rollout])

    def add_turns(self, rollouts: List[Tuple[ListModalityFrame, float]]):
        """
        增加一轮消息（含重新生成）
        """
        self.turns.append(rollouts)

    def supplement_at_turn(
        self, turn: int, rollouts: List[Tuple[ListModalityFrame, float]]
    ):
        """
        再某一轮补充重新生成的消息
        """
        self.turns[turn] += rollouts

    def get_chosen_history(
        self, only_get_untrained=False
    ) -> List[ListModalityFrame]:
        """
        获取真实对话历史（不含重新生成和分数）
        """
        out_turns = [
            turn[-1]
            for turn in (
                self.turns
                if not only_get_untrained
                else self.turns[self.untrained_history_pointer :]
            )
        ]
        out_turns_without_score= [
            turn[0] for turn in out_turns
        ]
        return out_turns_without_score


    def choose(self, index: int):
        """
        选择合适的rollout作为历史
        """
        if not self.turns:
            return
        assert len(self.turns[-1] > index)
        item = self.turns[-1].pop(index)
        self.turns[-1].append(item)
        
        
    # =================================================
    # ===================对话模式工具===================
    # =================================================
    def back_to_last(self):
        """
        回退到上一轮
        """
        self.turns.pop(-1)

