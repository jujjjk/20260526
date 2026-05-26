#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from collections import defaultdict

import requests


MOTOR_ORDER = [
    0x11, 0x12, 0x13,
    0x21, 0x22, 0x23,
    0x31, 0x32, 0x33,
    0x41, 0x42, 0x43,
]


def fetch_state(base_url: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/api/state"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def state_for(raw: dict, motor_id: int) -> dict:
    return raw.get(hex(motor_id)) or raw.get(str(motor_id)) or {}


def board_name(tag: int) -> str:
    if tag == 0xA1:
        return "A"
    if tag == 0xB1:
        return "B"
    return "?"


def summarize(raw: dict, prev_by_id: dict | None, stale_ms: float) -> tuple[str, dict]:
    lines = []
    current_by_id = {}
    boards = defaultdict(list)

    for mid in MOTOR_ORDER:
        item = state_for(raw, mid)
        current_by_id[mid] = item
        boards[int(item.get("board_tag", 0))].append((mid, item))

    for tag in sorted(boards):
        items = boards[tag]
        seqs = {int(item.get("snapshot_seq", -1)) for _, item in items}
        ticks = {int(item.get("board_tick_ms", -1)) for _, item in items}
        board_label = board_name(tag)
        seq_text = ",".join(str(x) for x in sorted(seqs))
        tick_text = ",".join(str(x) for x in sorted(ticks))
        lines.append(f"Board {board_label} tag=0x{tag:02X} seq={seq_text} tick_ms={tick_text}")

        for mid, item in items:
            online = bool(item.get("online", False))
            age_ms = float(item.get("age_ms", 999999))
            angle = float(item.get("angle", 0.0))
            speed = float(item.get("speed", 0.0))
            prev = prev_by_id.get(mid, {}) if prev_by_id else {}
            changed = ""
            if prev:
                if (
                    angle != float(prev.get("angle", angle))
                    or speed != float(prev.get("speed", speed))
                    or int(item.get("snapshot_seq", -1)) != int(prev.get("snapshot_seq", -1))
                ):
                    changed = "*"

            state = "OK"
            if not online:
                state = "OFFLINE"
            elif age_ms > stale_ms:
                state = "STALE"

            lines.append(
                f"  0x{mid:02X} {state:7s} age={age_ms:7.1f}ms "
                f"angle={angle:+8.4f} speed={speed:+8.4f} {changed}"
            )

    diagnoses = []
    for tag, items in boards.items():
        seqs_now = [int(item.get("snapshot_seq", -1)) for _, item in items]
        seqs_prev = [
            int((prev_by_id or {}).get(mid, {}).get("snapshot_seq", -2))
            for mid, _ in items
        ]
        seq_changed = seqs_now != seqs_prev
        any_online = any(bool(item.get("online", False)) for _, item in items)
        all_old = all(float(item.get("age_ms", 999999)) > stale_ms for _, item in items)

        label = board_name(tag)
        if prev_by_id and not seq_changed:
            diagnoses.append(f"Board {label}: snapshot_seq is not changing; check SPI target/device wiring.")
        elif not any_online or all_old:
            diagnoses.append(f"Board {label}: SPI is alive, but CAN feedback is stale/offline.")

    if diagnoses:
        lines.append("Diagnosis:")
        lines.extend(f"  - {d}" for d in diagnoses)

    return "\n".join(lines), current_by_id


def main():
    parser = argparse.ArgumentParser(description="Poll NX motor state API and diagnose stale motor feedback.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--period", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--stale-ms", type=float, default=100.0)
    args = parser.parse_args()

    prev = None
    while True:
        try:
            raw = fetch_state(args.base_url, args.timeout)
            text, prev = summarize(raw, prev, args.stale_ms)
            print("\033[2J\033[H", end="")
            print(text)
        except Exception as exc:
            print(f"ERROR: {exc}")
        time.sleep(args.period)


if __name__ == "__main__":
    main()
