"""Browser command-contract checks with a local, simulated v3/v4 live API.

This verifies browser safety and command payloads; it does not evaluate PPO.
Run: .venv/bin/python tests/verify_keyboard_web.py
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    contract = json.loads((ROOT / "config/robot_contract.json").read_text())
    state = {
        "controller_version": "q1_leg_native_v3", "ready": False,
        "skill": "sidestep", "status": "running", "mode": "stand",
        "sequence_active": False, "duration": 10,
        "command": {"vx": 0, "vy": 0, "wz": 0},
        "frame": {"time": 0, "q": contract["default"],
                  "root": [0, 0, contract["root_height"]],
                  "quaternion": [1, 0, 0, 0]},
    }
    commands, errors = [], []
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietStaticHandler, directory=str(ROOT)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path="/opt/google/chrome/chrome", headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle",
                      "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/status", lambda route: route.fulfill(json={"skills": {}}))
            page.route("**/api/live", lambda route: route.fulfill(json=state))

            def command(route):
                payload = route.request.post_data_json
                commands.append(payload)
                if payload["action"] == "velocity":
                    state["command"] = {k: payload[k] for k in ("vx", "vy", "wz")}
                    state["mode"] = "walk" if any(state["command"].values()) else "stand"
                elif payload["action"] == "sidestep_sequence":
                    state.update(sequence_active=True, mode="sidestep_left")
                elif payload["action"] == "stop":
                    state.update(sequence_active=False, mode="stand",
                                 command={"vx": 0, "vy": 0, "wz": 0})
                route.fulfill(json={"ok": True})

            page.route("**/api/live/command", command)
            page.goto(f"http://127.0.0.1:{server.server_port}/web/")
            page.wait_for_function("document.querySelector('#status').textContent.includes('已擷取')")
            page.keyboard.press("w")
            assert not commands, "Reference playback sent a live command"
            assert page.locator("#keyboard-toggle").is_disabled()

            page.locator("#source").select_option("live")
            page.wait_for_function("document.querySelector('#keyboard-status').textContent.includes('舊版')")
            assert page.locator("#keyboard-toggle").is_disabled()
            assert page.locator("#sidestep-sequence").is_disabled()
            page.keyboard.press("w")
            assert not any(c["action"] == "velocity" for c in commands)

            state["controller_version"] = "q1_leg_v4"
            page.wait_for_function("document.querySelector('#keyboard-status').textContent.includes('尚未就緒')")
            assert page.locator("#sidestep-sequence").is_disabled()
            state["ready"] = True
            page.wait_for_function("!document.querySelector('#keyboard-toggle').disabled")
            page.locator("#canvas canvas").focus()
            commands.clear()
            page.keyboard.press("w")
            assert not commands, "Movement was sent before keyboard activation"

            page.locator("#keyboard-toggle").click()
            page.keyboard.down("w")
            page.wait_for_timeout(180)
            assert commands[-1] == {"action": "velocity", "vx": .22, "vy": 0, "wz": 0}
            assert sum(c["action"] == "velocity" for c in commands) >= 2, "No held-key heartbeat"
            page.keyboard.down("d")
            page.wait_for_timeout(100)
            assert commands[-1]["vx"] == .22 and commands[-1]["vy"] == -.18
            page.keyboard.up("w")
            page.wait_for_timeout(100)
            assert commands[-1]["vx"] == 0 and commands[-1]["vy"] == -.18
            page.keyboard.up("d")
            page.wait_for_timeout(100)
            assert commands[-1] == {"action": "velocity", "vx": 0, "vy": 0, "wz": 0}
            page.keyboard.down("q")
            page.wait_for_timeout(100)
            assert commands[-1]["wz"] == .6
            page.keyboard.up("q")

            # Native form input must retain its own key behavior.
            page.evaluate("document.body.insertAdjacentHTML('beforeend','<input id=typing-test aria-label=test>')")
            page.locator("#typing-test").focus()
            page.wait_for_timeout(100)
            count = len(commands)
            page.keyboard.type("wasdqe ")
            page.wait_for_timeout(160)
            assert page.locator("#typing-test").input_value() == "wasdqe "
            assert len(commands) == count, "Typing in a field sent a command"
            page.locator("#typing-test").evaluate("e=>e.remove()")

            page.locator("#canvas canvas").focus()
            page.keyboard.down("w")
            page.wait_for_timeout(100)
            page.evaluate("window.dispatchEvent(new Event('blur'))")
            page.wait_for_timeout(100)
            assert commands[-1]["action"] == "stop"
            assert page.locator("#keyboard-toggle").get_attribute("aria-pressed") == "false"
            page.keyboard.up("w")
            count = len(commands)
            page.wait_for_timeout(250)
            assert len(commands) == count, "Movement kept streaming after blur"

            # Press-and-hold buttons use the same velocity contract and release to zero.
            button = page.locator('[data-control-key="KeyW"]')
            button.scroll_into_view_if_needed()
            box = button.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.wait_for_timeout(100)
            assert commands[-1]["vx"] == .22
            page.mouse.up()
            page.wait_for_timeout(100)
            assert commands[-1]["vx"] == 0

            # A sequence locks movement, and Space can abort even with keyboard unarmed.
            page.locator("#keyboard-toggle").click()
            assert page.locator("#keyboard-toggle").get_attribute("aria-pressed") == "false"
            page.locator("#sidestep-sequence").click()
            page.wait_for_function("document.querySelector('#sidestep-sequence').disabled")
            assert commands[-1] == {"action": "sidestep_sequence"}
            count = len(commands)
            page.keyboard.press("w")
            page.wait_for_timeout(160)
            assert len(commands) == count
            assert page.locator('[data-control-key="KeyW"]').is_disabled()
            page.keyboard.press("Space")
            page.wait_for_timeout(100)
            assert commands[-1] == {"action": "stop"}
            page.wait_for_function("!document.querySelector('#sidestep-sequence').disabled")

            page.locator("#keyboard-toggle").click()
            page.keyboard.down("a")
            page.wait_for_timeout(100)
            assert commands[-1]["vy"] == .18
            page.locator("#source").select_option("reference")
            page.wait_for_timeout(120)
            assert commands[-1]["action"] == "stop"
            page.keyboard.up("a")
            assert page.locator("#keyboard-toggle").is_disabled()
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print("Keyboard web verified: v4 readiness gate, live-only commands, WASD/QE, heartbeat, "
          "keyup/blur stop, input focus, pointer release, sequence lock, Space abort and source switch.")


if __name__ == "__main__":
    main()
