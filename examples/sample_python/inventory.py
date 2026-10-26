"""示例目标：一个包含分支、异常和可变状态的小模块（MIT，随仓库分发）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Inventory:
    """库存：支持入库、出库与低库存判断。"""

    items: dict[str, int] = field(default_factory=dict)
    low_threshold: int = 3

    def add(self, name: str, quantity: int) -> int:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not name:
            raise ValueError("name must not be empty")
        self.items[name] = self.items.get(name, 0) + quantity
        return self.items[name]

    def remove(self, name: str, quantity: int) -> int:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if name not in self.items:
            raise KeyError(name)
        if self.items[name] < quantity:
            raise ValueError("insufficient stock")
        self.items[name] -= quantity
        if self.items[name] == 0:
            del self.items[name]
        return self.items.get(name, 0)

    def low_stock(self) -> list[str]:
        return sorted(name for name, count in self.items.items() if count <= self.low_threshold)

    def total(self) -> int:
        return sum(self.items.values())


def normalize_name(raw: str) -> str:
    """规范化名称：去空格、转小写、把连续空白压成一个下划线。"""
    cleaned = " ".join(raw.strip().lower().split())
    return cleaned.replace(" ", "_")


def summarize(counts: list[int]) -> dict[str, float]:
    """返回总数、均值与最大值；空列表时返回 0。"""
    if not counts:
        return {"total": 0, "mean": 0.0, "max": 0}
    total = sum(counts)
    return {"total": total, "mean": total / len(counts), "max": max(counts)}
