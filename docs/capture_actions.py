"""Capture a public GitHub Actions job page for the report.

    python docs/capture_actions.py <job-url> [output.png]
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

DEFAULT_URL = (
    "https://github.com/earnest-s/ChurnGuard_AI/actions/runs/"
    "40449950182/job/122010080636"
)
OUT = Path(__file__).resolve().parent / "report_assets" / "github_actions.png"


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1280,820")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(5)

        # Dismiss a possible cookie banner without touching the run itself.
        for label in ("Accept all", "Accept", "Reject all"):
            buttons = driver.find_elements(By.XPATH, f"//button[contains(., '{label}')]")
            if buttons:
                buttons[0].click()
                time.sleep(1)
                break

        title = driver.title
        body = driver.find_element(By.TAG_NAME, "body").text
        driver.save_screenshot(str(out.with_name("github_actions_full.png")))

        # Focus the job panel: job name, success status and the step list.
        clip = {"x": 330, "y": 370, "width": 920, "height": 430, "scale": 2}
        shot = driver.execute_cdp_cmd(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": True, "clip": clip},
        )
        out.write_bytes(base64.b64decode(shot["data"]))
        print(f"title: {title}")
        print(f"wrote {out} ({out.stat().st_size // 1024} KiB) and full page")
        for line in body.splitlines():
            if line.strip():
                print(line[:140])
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
