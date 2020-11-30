from typing import Tuple, List, Dict, Any


class TrajectoryItem:
    def __init__(self, intention: Tuple[int, ...], positions: List[Tuple[float, float]], delta: float = 0.1):
        self.intention = intention
        self.positions = positions
        self.delta = delta

    def to_dict(self) -> Dict[str, Any]:
        return {'intention': self.intention,
                'delta': self.delta,
                'positions': self.positions}

    def to_str(self) -> str:
        intention_str = '{:d},{:d}'.format(self.intention[0], self.intention[1])
        position_str = ','.join(['{:.2f},{:.2f}'.format(*pos) for pos in self.positions])
        return ','.join([intention_str, position_str])

    def __str__(self):
        return 'intention: ({:d}, {:d}), items: {:05d}, delta: {:.4f}'.format(
            self.intention[0], self.intention[1], len(self.positions), self.delta)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.positions)

    @staticmethod
    def parse_str(line: str):
        words = line.split(',')
        intention = tuple([int(w) for w in words[:2]])
        values = [float(v) for v in words[2:]]
        positions = list(zip(values[::2], values[1::2]))
        return TrajectoryItem(intention, positions)
