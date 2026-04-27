from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RiskManager:
    bankroll: float = 1_000.0
    max_default_streak: int = 3
    streaks: Dict[str, int] = field(default_factory=dict)

    def compute_kelly(self, win_prob: float, odds: float) -> float:
        win_prob = max(0.0, min(1.0, float(win_prob)))
        odds = max(1e-9, float(odds))
        b = odds - 1.0
        q = 1.0 - win_prob
        if b <= 0:
            return 0.0
        kelly = (b * win_prob - q) / b
        return max(0.0, kelly)

    def update_streak(self, signal_name: str, won: bool) -> int:
        current = int(self.streaks.get(signal_name, 0))
        if won:
            current = 0
        else:
            current += 1
        self.streaks[signal_name] = current
        return current

    def should_suspend(self, signal_name: str, max_allowed: int = 3) -> bool:
        limit = int(max_allowed) if max_allowed is not None else self.max_default_streak
        return int(self.streaks.get(signal_name, 0)) >= limit

    def get_bet_recommendation(
        self,
        signal_name: str,
        hit_rate: float,
        odds: float,
        bankroll: float,
        max_allowed: int = 3,
    ) -> Dict[str, float | bool | str]:
        suspended = self.should_suspend(signal_name, max_allowed=max_allowed)
        kelly = self.compute_kelly(hit_rate, odds)
        half_kelly = 0.5 * kelly * float(bankroll)
        if suspended:
            half_kelly = 0.0
        return {
            "signal": signal_name,
            "suspended": suspended,
            "kelly_fraction": kelly,
            "recommended_stake": max(0.0, half_kelly),
            "odds": float(odds),
            "hit_rate": float(hit_rate),
        }
