"""Seam tests: fluent open-source directing (no large head-yaw)."""
from __future__ import annotations

import unittest

from llm_director import (
    boost_i2v_motion_prompt,
    compose_i2v_prompt,
    is_lookback_beat_action,
    prefers_face_micro_motion,
    sanitize_shot_motion,
    validate_lookback_budget,
)


TOPIC = (
    "赏金猎人走入暗巷。微型的神秘无人机从天而降，猎人察觉后猛然回头"
)


class SanitizeFaceSafeDirecting(unittest.TestCase):
    def test_large_turn_becomes_overshoulder_micro(self):
        out = sanitize_shot_motion(
            "迅速转身，面向无人机",
            context_text=TOPIC,
            previous=["向纵深走入暗巷", "过肩侧脸微动，目光警觉",
                      "镜头推向主体背部，微型的神秘无人机从画面上方进入"],
            index=4,
        )
        self.assertFalse(re_search_large_turn(out))
        self.assertIn("微动", out)
        self.assertIn("无人机", out)

    def test_lookback_canonicalized_to_micro(self):
        out = sanitize_shot_motion(
            "猛然回头看向声响",
            context_text=TOPIC,
            previous=["向纵深走入暗巷"],
            index=1,
        )
        self.assertEqual(out, "过肩侧脸微动，目光警觉")
        self.assertTrue(prefers_face_micro_motion(out, TOPIC))

    def test_climax_enter_preserved(self):
        action = "镜头推向主体背部，微型的神秘无人机从画面上方进入"
        out = sanitize_shot_motion(
            action,
            context_text=TOPIC,
            previous=["向纵深走入暗巷", "过肩侧脸微动，目光警觉"],
            index=2,
        )
        self.assertIn("无人机", out)
        self.assertIn("进入", out)
        self.assertFalse(prefers_face_micro_motion(out, TOPIC))

    def test_second_lookback_not_kept(self):
        out = sanitize_shot_motion(
            "过肩回望，眼神锐利",
            context_text=TOPIC,
            previous=["向纵深走入暗巷", "过肩侧脸微动，目光警觉"],
            index=2,
        )
        self.assertFalse(is_lookback_beat_action(out) and "回望" in out)
        # after one lookback without enter yet → enter preferred
        self.assertTrue("进入" in out or "注视" in out or "锁定" in out)

    def test_notice_look_up_stays_rear_no_face_invent(self):
        out = sanitize_shot_motion(
            "突然停下脚步，抬头望向上方",
            context_text=TOPIC,
            previous=["向纵深走入暗巷"],
            index=1,
        )
        self.assertIn("背影", out)
        self.assertNotIn("转头", out)
        self.assertFalse(prefers_face_micro_motion(out, TOPIC))


class ComposeAndBoost(unittest.TestCase):
    def test_compose_keeps_drone_enter(self):
        prev = ["向纵深走入暗巷", "过肩侧脸微动，目光警觉"]
        action = "镜头推向主体背部，微型的神秘无人机从画面上方进入"
        composed = compose_i2v_prompt(
            "黑色风衣赏金猎人", action, 2, context_text=TOPIC, previous=prev
        )
        self.assertIn("无人机", composed)
        boosted = boost_i2v_motion_prompt(
            action, rear_pushin=True, context_text=TOPIC
        )
        self.assertIn("进入", boosted)

    def test_face_micro_boost_forbids_large_yaw(self):
        text = "过肩侧脸微动，目光警觉"
        boosted = boost_i2v_motion_prompt(text, context_text=TOPIC)
        self.assertIn("定妆", boosted)
        self.assertIn("禁止", boosted)
        self.assertTrue(prefers_face_micro_motion(text, TOPIC))


class LookbackBudget(unittest.TestCase):
    def test_two_lookbacks_rejected(self):
        beats = [
            {"action": "向纵深走入暗巷"},
            {"action": "过肩侧脸微动，目光警觉"},
            {"action": "过肩回望，眼神锐利"},
        ]
        ok, reason = validate_lookback_budget(beats)
        self.assertFalse(ok)
        self.assertIn("look-back", reason)


def re_search_large_turn(text: str) -> bool:
    import re

    return bool(
        re.search(
            r"迅速转身|猛然回头|猛然回望|全身急转|大幅度",
            text or "",
        )
    )


if __name__ == "__main__":
    unittest.main()
