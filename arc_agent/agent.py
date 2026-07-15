import argparse
import os
import sys
import logging
import io

from anthropic import Anthropic
from arc_agi import Arcade
from arcengine import GameAction

# Fix Windows console encoding for frame characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)

# ── LLM config (from settings.json env) ──────────────────────
API_KEY = os.getenv("ARC_AGENT_API_KEY")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
MODEL = os.getenv("ANTHROPIC_MODEL", "DeepSeek-V4-Pro[1m]")

# ── Color map ────────────────────────────────────────────────
COLOR_LABELS = {
    0: "white", 1: "off-white", 2: "neutral-light", 3: "neutral",
    4: "off-black", 5: "black", 6: "magenta", 7: "magenta-light",
    8: "red", 9: "blue", 10: "blue-light", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}

COLOR_CHAR = {
    0: "·", 1: "░", 2: "▒", 3: "▓", 4: "▌", 5: "█",
    6: "M", 7: "m", 8: "R", 9: "B", 10: "b",
    11: "Y", 12: "O", 13: "r", 14: "G", 15: "P",
}


def frame_to_text(frame) -> str:
    lines = []
    for row in frame:
        lines.append("".join(COLOR_CHAR.get(int(v), "?") for v in row))
    return "\n".join(lines)


def get_cropped_frame_text(frame) -> str:
    import numpy as np
    frame_arr = np.array(frame)
    # Find active elements (non-zero/non-background)
    active_indices = np.argwhere(frame_arr != 0)
    if active_indices.size == 0:
        return "Empty Grid (all 0/white)"
    
    r_indices, c_indices = active_indices[:, 0], active_indices[:, 1]
    r_min, r_max = r_indices.min(), r_indices.max()
    c_min, c_max = c_indices.min(), c_indices.max()
    
    # Add a padding of 3 cells for spatial context
    padding = 3
    r_start = max(0, r_min - padding)
    r_end = min(63, r_max + padding)
    c_start = max(0, c_min - padding)
    c_end = min(63, c_max + padding)
    
    cropped = frame_arr[r_start:r_end+1, c_start:c_end+1]
    
    lines = []
    for r_idx, row in enumerate(cropped):
        row_str = "".join(COLOR_CHAR.get(int(v), "?") for v in row)
        lines.append(f"Row {r_start + r_idx:02d}: {row_str}")
        
    cols_header = "        " + "".join(str((c_start + i) % 10) for i in range(c_end - c_start + 1))
    
    subgrid_text = (
        f"Active bounding box: Rows [{r_start}, {r_end}], Columns [{c_start}, {c_end}]\n"
        f"{cols_header}\n" + "\n".join(lines)
    )
    return subgrid_text


def format_legend() -> str:
    items = [f"{c}={COLOR_LABELS[d]}" for d, c in sorted(COLOR_CHAR.items())]
    return "  ".join(items)


def action_name(action_id: int) -> str:
    try:
        return GameAction.from_id(action_id).name
    except ValueError:
        return f"UNKNOWN({action_id})"


# ── Global game state ───────────────────────────────────────
arc: Arcade | None = None
env = None
agent_memory_str = "No memory stored yet. Use save_memory tool to record discovered rules and mappings."


def tool_observe(show_full_grid: bool = False) -> str:
    global env
    if env is None:
        return "Error: no game loaded."
    obs = env.observation_space
    if obs is None:
        return "No observation available."

    frames = obs.frame
    if frames:
        if show_full_grid:
            frame_text = frame_to_text(frames[0])
        else:
            frame_text = get_cropped_frame_text(frames[0])
    else:
        frame_text = "(no frame data)"
        
    available = [action_name(a) for a in obs.available_actions]

    return (
        f"State: {obs.state.name}\n"
        f"Levels: {obs.levels_completed}/{obs.win_levels}\n"
        f"Available actions: {', '.join(available)}\n"
        f"--- Frame ---\n{frame_text}\n"
        f"--- Legend ---\n{format_legend()}"
    )


def tool_act(action_id: int, reasoning: str = "", data: dict | None = None) -> str:
    global env
    if not (1 <= action_id <= 7):
        return f"Error: action_id must be 1-7, got {action_id}"

    action = GameAction.from_id(action_id)
    result = env.step(action, data=data or {}, reasoning=reasoning[:16000] if reasoning else None)
    if result is None:
        return "Error: step() returned None."

    frames = result.frame
    frame_text = get_cropped_frame_text(frames[0]) if frames else "(no frame data)"
    available = [action_name(a) for a in result.available_actions]

    return (
        f"Action: {action.name}\n"
        f"Result state: {result.state.name}\n"
        f"Levels: {result.levels_completed}/{result.win_levels}\n"
        f"Available actions now: {', '.join(available)}\n"
        f"--- Frame (Cropped) ---\n{frame_text}"
    )


def tool_reset() -> str:
    global env
    result = env.reset()
    if result is None:
        return "Error: reset failed."

    frames = result.frame
    frame_text = get_cropped_frame_text(frames[0]) if frames else "(no frame data)"
    available = [action_name(a) for a in result.available_actions]

    return (
        f"Game reset.\nState: {result.state.name}\n"
        f"Levels: {result.levels_completed}/{result.win_levels}\n"
        f"Available actions: {', '.join(available)}\n"
        f"--- Frame (Cropped) ---\n{frame_text}"
    )


def tool_save_memory(memory: str) -> str:
    global agent_memory_str
    agent_memory_str = memory
    return f"Memory successfully updated to:\n{agent_memory_str}"


TOOLS = [
    {
        "name": "observe",
        "description": "Observe the current game state: view the frame grid, game status, level progress, and available actions. By default, it returns a cropped bounding box of active elements to save tokens. Set show_full_grid to true to see the entire 64x64 grid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "show_full_grid": {
                    "type": "boolean",
                    "description": "Set to true to see the full 64x64 grid instead of the cropped bounding box.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "act",
        "description": "Perform a game action. Choose an action_id from the available_actions list shown by observe(). Use reasoning to explain your choice. For complex actions that require coordinates, include the data field.",
        "input_schema": {
            "type": "object",
            "required": ["action_id"],
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": "Action number (1-7). Must be one of the currently available_actions.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for why you chose this action.",
                },
                "data": {
                    "type": "object",
                    "description": "Extra data for complex actions (e.g. ACTION6 requires x,y coordinates on the 64x64 grid).",
                    "properties": {
                        "x": {"type": "integer", "minimum": 0, "maximum": 63},
                        "y": {"type": "integer", "minimum": 0, "maximum": 63},
                    },
                },
            },
        },
    },
    {
        "name": "reset",
        "description": "Reset the game to its initial state.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "save_memory",
        "description": "Save important rules, observations, or hypotheses you've discovered about the game. This memory will persist across steps and resets.",
        "input_schema": {
            "type": "object",
            "required": ["memory"],
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "The updated list of rules, mappings, and facts you want to remember.",
                }
            },
        },
    },
]

SYSTEM_PROMPT = """You are an ARC-AGI-3 game-playing agent. You control a character in a 64x64 grid world. Use tools to observe and act.

## Persisted Agent Memory
{AGENT_MEMORY}

## Tools
- **observe(show_full_grid)** — see the current frame, state, levels, and available actions. By default, it returns a cropped subgrid containing active objects to save tokens and improve readability. Set show_full_grid to true if you need the full 64x64 grid.
- **act(action_id, reasoning, data)** — perform an action. `action_id` is 1-7 (must be in available_actions). Add brief `reasoning`. Use `data` only for complex actions requiring coordinates.
- **reset()** — restart the game from the beginning.
- **save_memory(memory)** — save rules, action effects, or maps you discover so they persist in the prompt.

## Game Rules & Mechanics (Crucial for Solving)
1. **Objective**: Your goal is to clear all target slots (often black blocks `█` or specific slot patterns) in each level to advance.
2. **Match Requirement**: To step into a target slot and clear it, your character must match the target's required **Shape**, **Color**, and **Rotation**. If you don't match, the slot acts as a wall and blocks you.
3. **Change Pads**: There are special transformer pads on the grid that you can step on to cycle your properties:
   - **Shape Changer**: Cycles your character's shape.
   - **Color Changer** (often multi-colored area containing different pixel colors): Cycles your character's color.
   - **Rotation Changer**: Cycles your character's rotation (rotates your sprite).
4. **Step Items**: Small green or yellow items on the grid can recharge your step counter.
5. **Resets**: Avoid calling `reset()` unless you are completely stuck. All levels progress is reset to 0/7 on reset.

## Grid Representation & Coordinates
* The frame has row numbers on the left (e.g., `Row 12:`) and a column index header at the top showing the last digit of the column number (0-9).
* Use these numbers to find absolute coordinates (row index = y, column index = x) for coordinate-based actions.

## Grid Legend
· = 0=white   ░ = 1=off-white   ▒ = 2=neutral-lt   ▓ = 3=neutral
▌ = 4=off-black   █ = 5=black
M = 6=magenta   m = 7=magenta-lt   R = 8=red
B = 9=blue   b = 10=blue-lt
Y = 11=yellow   O = 12=orange   r = 13=maroon   G = 14=green   P = 15=purple

## Strategy
1. Call observe() first to locate your character, targets, and change pads.
2. Formulate a plan:
   - What shape, color, and rotation does the target require? (Observe the target sprite details or test entering it).
   - Find the path to the pads to change your shape, color, and rotation to match the target.
   - Once matched, navigate to the target slot and step on it to clear it.
3. Write down your current state and pad effects using `save_memory`.
4. Avoid calling `reset` unless absolutely stuck."""


def prune_messages(messages, max_turns):
    if len(messages) <= 1:
        return messages
    
    last_msg = messages[-1]
    if last_msg["role"] == "assistant":
        keep_count = 2 * max_turns + 1
    else:
        keep_count = 2 * max_turns
        
    keep_count = min(keep_count, len(messages) - 1)
    
    slice_start = len(messages) - keep_count
    while slice_start < len(messages) and messages[slice_start]["role"] != "assistant":
        slice_start += 1
        
    return [messages[0]] + messages[slice_start:]


def main():
    global arc, env

    if not API_KEY:
        raise RuntimeError("ARC_AGENT_API_KEY is not set (check settings.json env)")

    p = argparse.ArgumentParser(description="ARC-AGI-3 LLM Agent")
    p.add_argument("--game", default="ls20", help="Game ID (default: ls20)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--model", default=MODEL, help=f"Model (default: {MODEL})")
    p.add_argument("--render-terminal", action="store_true")
    args = p.parse_args()

    logging.getLogger("arc_agi").setLevel(logging.WARNING)

    render_mode = "terminal" if args.render_terminal else None
    arc = Arcade()
    env = arc.make(args.game, render_mode=render_mode)
    if env is None:
        print(f"Failed to load game: {args.game}")
        sys.exit(1)

    env.reset()

    client = Anthropic(api_key=API_KEY, base_url=BASE_URL)

    # Build initial observation
    obs = env.observation_space
    frame_text = ""
    if obs and obs.frame:
        frame_text = get_cropped_frame_text(obs.frame[0])
    available = [action_name(a) for a in obs.available_actions] if obs else []

    init_msg = (
        f"You are playing ARC-AGI-3 game '{args.game}'. "
        f"Start by observing the game state, then play step by step.\n\n"
        f"Initial state: {obs.state.name if obs else '?'}\n"
        f"Levels: {obs.levels_completed if obs else 0}/{obs.win_levels if obs else 0}\n"
        f"Available actions: {', '.join(available)}\n"
        f"--- Initial Frame ---\n{frame_text}"
    )

    messages = [{"role": "user", "content": init_msg}]

    steps = 0
    last_state = None

    while steps < args.max_steps:
        # Dynamically inject memory into system prompt
        system_prompt = SYSTEM_PROMPT.replace("{AGENT_MEMORY}", agent_memory_str)

        # Prune context history to keep token counts low and prevent API timeouts
        num_turns_to_keep = 3
        send_messages = prune_messages(messages, num_turns_to_keep)

        print(f"\n[Step {steps+1}] Calling LLM API...")
        import time
        t0 = time.time()
        response = client.messages.create(
            model=args.model,
            max_tokens=4096,
            system=system_prompt,
            messages=send_messages,
            tools=TOOLS,
        )
        print(f"[Step {steps+1}] LLM API returned in {time.time() - t0:.2f}s")

        # Extract text and tool_use blocks, preserve ALL blocks for history
        text_parts = []
        tool_uses = []
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_uses.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                assistant_content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
            elif block.type == "redacted_thinking":
                assistant_content.append({
                    "type": "redacted_thinking",
                    "data": block.data,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        for t in text_parts:
            print(f"[Agent] {t}")

        if not tool_uses:
            if last_state in ("WIN", "GAME_OVER"):
                break
            continue

        # Execute tools and collect results
        tool_results = []
        for tu in tool_uses:
            if tu.name == "observe":
                show_full = tu.input.get("show_full_grid", False)
                result = tool_observe(show_full)
            elif tu.name == "act":
                aid = tu.input.get("action_id", 1)
                reasoning = tu.input.get("reasoning", "")
                data = tu.input.get("data", {})
                result = tool_act(aid, reasoning, data)
                steps += 1
            elif tu.name == "reset":
                result = tool_reset()
            elif tu.name == "save_memory":
                memory_val = tu.input.get("memory", "")
                result = tool_save_memory(memory_val)
            else:
                result = f"Unknown tool: {tu.name}"

            # Print truncated result
            display = result
            marker = "--- Frame"
            idx = display.find(marker)
            if idx >= 0:
                display = display[:idx] + marker + "\n[grid omitted]"

            print(f"[Step {steps}] {tu.name}: {display[:200]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

        # Check game state
        obs = env.observation_space
        if obs:
            last_state = obs.state.name
        if last_state in ("WIN", "GAME_OVER"):
            print(f"\nGame ended: {last_state}")
            break

    scorecard = arc.get_scorecard()
    if scorecard:
        sc = scorecard.model_dump_json(indent=2)
        print(f"\nScorecard:\n{sc[:2000]}")


if __name__ == "__main__":
    main()
