"""Capture the report screenshots from the running local stack.

Prerequisites:
    backend:  uvicorn main:app --host 127.0.0.1 --port 8000
    frontend: VITE_API_URL=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1

Run from the repository root:
    backend\\.venv\\Scripts\\python.exe docs/capture_screenshots.py
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(exist_ok=True)


def _driver(width: int = 760, height: int = 1100) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return webdriver.Chrome(options=options)


def _full_page(driver: webdriver.Chrome, path: Path) -> None:
    metrics = driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
    size = metrics["cssContentSize"]
    shot = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {
            "format": "png",
            "captureBeyondViewport": True,
            "clip": {
                "x": 0,
                "y": 0,
                "width": size["width"],
                "height": size["height"],
                "scale": 1,
            },
        },
    )
    path.write_bytes(base64.b64decode(shot["data"]))
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")


def _result_clip(driver: webdriver.Chrome) -> dict | None:
    """Bounding box from the verdict panel through the SHAP force plot."""
    return driver.execute_script(
        """
        const start = document.querySelector('.result');
        const end = document.querySelector('.force-plot');
        if (!start || !end) return null;
        const sr = start.getBoundingClientRect();
        const er = end.getBoundingClientRect();
        const left = Math.min(sr.left, er.left) + window.scrollX;
        const right = Math.max(sr.right, er.right) + window.scrollX;
        const top = sr.top + window.scrollY;
        const bottom = er.bottom + window.scrollY;
        return {
          x: Math.max(0, left - 8),
          y: Math.max(0, top - 8),
          width: right - left + 16,
          height: bottom - top + 16,
        };
        """
    )


def _clip_shot(driver: webdriver.Chrome, path: Path, clip: dict) -> None:
    shot = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {
            "format": "png",
            "captureBeyondViewport": True,
            "clip": {**clip, "scale": 2},
        },
    )
    path.write_bytes(base64.b64decode(shot["data"]))
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB, {clip['width']}x{clip['height']} css)")


def capture_gui() -> None:
    driver = _driver()
    try:
        driver.get("http://127.0.0.1:5173/")
        wait = WebDriverWait(driver, 30)

        wait.until(EC.presence_of_element_located((By.NAME, "tenure")))
        tenure = driver.find_element(By.NAME, "tenure")
        tenure.clear()
        tenure.send_keys("2")

        charges = driver.find_element(By.NAME, "monthly_charges")
        charges.clear()
        charges.send_keys("95")

        calls = driver.find_element(By.NAME, "support_calls")
        calls.clear()
        calls.send_keys("4")

        # Open the optional profile and supply the fields that matter most.
        driver.find_element(By.CSS_SELECTOR, "details.advanced summary").click()
        time.sleep(0.2)
        Select(driver.find_element(By.NAME, "internet_service")).select_by_value("Fiber optic")
        Select(driver.find_element(By.NAME, "payment_method")).select_by_value("Electronic check")
        Select(driver.find_element(By.NAME, "paperless_billing")).select_by_value("Yes")
        Select(driver.find_element(By.NAME, "tech_support")).select_by_value("No")

        driver.find_element(By.CSS_SELECTOR, "button[type=submit]").click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".force-plot")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".result")))

        verdict = driver.find_element(By.CSS_SELECTOR, ".result h2").text
        probability = driver.find_element(By.CSS_SELECTOR, ".result p").text
        print(f"GUI prediction: {verdict} | {probability}")

        _full_page(driver, ASSETS / "gui_full_page.png")
        clip = _result_clip(driver)
        if not clip:
            raise RuntimeError("Could not locate the result panel")
        _clip_shot(driver, ASSETS / "gui_prediction.png", clip)
    finally:
        driver.quit()


def capture_metrics() -> None:
    driver = _driver(width=1280, height=900)
    try:
        driver.get("http://127.0.0.1:8000/metrics")
        time.sleep(0.5)
        driver.save_screenshot(str(ASSETS / "metrics_endpoint.png"))
        body = driver.find_element(By.TAG_NAME, "body").text
        (ASSETS / "metrics_excerpt.txt").write_text(body[:6000], encoding="utf-8")
        print(f"wrote {ASSETS / 'metrics_endpoint.png'} ({body[:120]!r}...)")
    finally:
        driver.quit()


if __name__ == "__main__":
    capture_gui()
    capture_metrics()
