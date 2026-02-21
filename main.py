import uvicorn
import webbrowser
import socket
import time
from threading import Thread


def wait_for_server(port: int, timeout: int = 30):
    """
    Polls the port until it's active, then opens the browser.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                print(f"[INFO] Server is UP on port {port}. Opening browser...")
                time.sleep(0.5)
                webbrowser.open(f"http://127.0.0.1:{port}/")   # ← opens the UI, not /docs
                return
        time.sleep(1)
    print("[ERROR] Server startup timed out.")


def launch_server():
    print("=" * 60)
    print("MEDGEMMA AI PSYCHIATRIST: SYSTEM LAUNCHER")
    print("=" * 60)

    port = 8000

    monitor_thread = Thread(target=wait_for_server, args=(port,))
    monitor_thread.daemon = True
    monitor_thread.start()

    print(f"[INFO] Booting FastAPI and loading Agents (MedGemma, RAG, etc.)...")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    launch_server()
