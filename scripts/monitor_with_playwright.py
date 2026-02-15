import asyncio
import os
import time
import subprocess
from playwright.async_api import async_playwright

async def monitor_dashboard():
    # 1. Start Streamlit Dashboard in background
    print("[*] Launching Streamlit Dashboard...")
    dashboard_process = subprocess.Popen(
        ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for dashboard to start
    await asyncio.sleep(5)

    os.makedirs("monitor_snapshots", exist_ok=True)

    async with async_playwright() as p:
        # 2. Launch Browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("[*] Navigating to Dashboard...")
        try:
            await page.goto("http://localhost:8501", timeout=60000)
            await page.wait_for_load_state("networkidle")
        except Exception as e:
            print(f"[!] Failed to load dashboard: {e}")
            dashboard_process.terminate()
            return

        # 3. Select 'Training Monitor' (Assuming it's a radio button)
        # Streamlit radio buttons are often inside specific divs. 
        # We look for the label "Training Monitor"
        try:
            print("[*] Switching to Training Monitor tab...")
            # Click the radio button for 'Training Monitor'
            # Note: Selectors might need adjustment based on Streamlit version
            await page.get_by_text("Training Monitor").click()
            await asyncio.sleep(2)
            
            # Enable Auto Refresh
            print("[*] Enabling Auto Refresh...")
            await page.get_by_text("Auto Refresh").click()
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"[!] Interaction failed: {e}")

        # 4. Loop for screenshots
        print("[*] Starting monitoring loop (Press Ctrl+C to stop)...")
        try:
            for i in range(10): # Take 10 snapshots then exit (demo mode)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"monitor_snapshots/status_{timestamp}.png"
                await page.screenshot(path=filename, full_page=True)
                print(f"    [+] Saved snapshot: {filename}")
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\n[*] Stopping monitoring...")

        await browser.close()
    
    dashboard_process.terminate()
    print("[*] Monitoring finished.")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_dashboard())
    except KeyboardInterrupt:
        pass
