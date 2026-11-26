"""选手只需修改这个文件。保持初始姿态的最小接口示例，不保证站立或行走。"""


class Policy:
    def __init__(self, player_id=0, config=None):
        self.player_id = player_id
        self.reset_count = -1
        self.targets = None

    def act(self, observation):
        # 复位后重新读取姿态；这里可以换成自己的感知、决策和运控代码。
        if observation["reset_count"] != self.reset_count:
            self.reset_count = observation["reset_count"]
            self.targets = observation["proprio"]["q"].copy()
        return {"joint_targets": self.targets.copy(), "fire": False}
