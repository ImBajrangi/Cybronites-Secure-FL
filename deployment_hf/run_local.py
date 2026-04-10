import subprocess
import time
import os
import signal
import sys
import urllib.request

# AI GUARDIAN | LOCAL ORCHESTRATION COORDINATOR
# Run this ONE file to start everything:
#   1. FastAPI Bridge (Dashboard WebSocket server)
#   2. Flower FL Server (Orchestrator)
#   3. Federated Clients (x2)

# Dynamically detect port from environment (HF default is 7860, local default 7880)
BRIDGE_PORT = int(os.environ.get("PORT", 7880))
FLOWER_PORT = int(os.environ.get("FLOWER_PORT", 8095))
ROUNDS = int(os.environ.get("ROUNDS", 5))

processes = []

def signal_handler(sig, frame):
    print("\nAI GUARDIAN | SHUTTING DOWN STACK...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def wait_for_bridge(port, timeout=15):
    """Wait until the bridge HTTP endpoint is reachable."""
    for i in range(timeout):
        try:
            # Use /api/health as defined in bridge.py
            r = urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=1)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    print("=" * 60)
    print("  AI GUARDIAN | SECURE FEDERATED LEARNING ORCHESTRATOR")
    print("=" * 60)
    
    cwd = os.getcwd()
    
    # Environment setup
    env = os.environ.copy()
    # PYTHONPATH should be project root to find Cybronites package
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    env["PORT"] = str(BRIDGE_PORT)
    env["FLOWER_PORT"] = str(FLOWER_PORT)

    # Update .env.local for dashboard auto-discovery while preserving other keys
    try:
        env_path = os.path.join(cwd, "dashboard", ".env.local")
        content = ""
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                content = f.read()
        
        # Regex update or append VITE_BACKEND_PORT
        import re
        port_pattern = r"VITE_BACKEND_PORT=\d+"
        new_line = f"VITE_BACKEND_PORT={BRIDGE_PORT}"
        
        if re.search(port_pattern, content):
            new_content = re.sub(port_pattern, new_line, content)
        else:
            new_content = content.strip() + f"\n{new_line}\n"
            
        with open(env_path, "w") as f:
            f.write(new_content)
        print(f"  [SYNC] Dashboard .env.local -> Port {BRIDGE_PORT} (Other keys preserved)")
    except Exception as e:
        print(f"  [WARN] Could not update .env.local: {e}")

    # Find Python interpreter
    paths_to_check = [
        os.path.join(cwd, "Cybronites/venv_mac/bin/python3"),
        os.path.join(cwd, ".venv/bin/python3"),
    ]
    python_path = sys.executable
    for p in paths_to_check:
        if os.path.exists(p):
            python_path = p
            break
    print(f"  [PYTHON] {python_path}")
    
    # ─── Cloud-Safe Logging Setup ───
    # Hugging Face Spaces run as a non-root user (UID 1000) with limited write access.
    # We use /tmp for logs to ensure 100% permission stability.
    is_cloud = os.environ.get("SPACE_ID") is not None or os.environ.get("HF_SPACE") == "true"
    log_path = "/tmp/backend.log" if is_cloud else "backend.log"
    json_path = "/tmp/backend.json" if is_cloud else "backend.json"
    
    print(f"\n  [1/3] Starting Bridge (:{BRIDGE_PORT}) + Flower Server (:{FLOWER_PORT})...")
    print(f"  [LOG] Redirecting logs to {log_path}")
    
    log_file = open(log_path, "w")
    # Initialize/Clear JSON log
    with open(json_path, "w") as f:
        f.write("") 
    
    server_proc = subprocess.Popen(
        [python_path, "-m", "Cybronites.server.server", "--flower_port", str(FLOWER_PORT), "--rounds", str(ROUNDS)],
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    processes.append(server_proc)

    def stream_logs(proc, file):
        for line in iter(proc.stdout.readline, ""):
            print(f"  [SERVER] {line.strip()}")
            file.write(line)
            file.flush()

    import threading
    log_thread = threading.Thread(target=stream_logs, args=(server_proc, log_file), daemon=True)
    log_thread.start()

    # 2. Wait for bridge to be ready (not just a sleep)
    print("  [WAIT] Waiting for bridge to come online...", end="", flush=True)
    if wait_for_bridge(BRIDGE_PORT):
        print(" READY ✓")
    else:
        print(" TIMEOUT (continuing anyway)")

    # 3. Launch Clients
    num_clients = 2
    for i in range(num_clients):
        print(f"  [2/3] Starting Client {i}...")
        client_proc = subprocess.Popen(
            [python_path, "-m", "Cybronites.client.client", str(i), str(num_clients)],
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(client_proc)
        
        def stream_client_logs(proc, index, file):
            for line in iter(proc.stdout.readline, ""):
                print(f"  [CLIENT {index}] {line.strip()}")
                file.write(f"  [CLIENT {index}] {line.strip()}\n")
                file.flush()
        
        threading.Thread(target=stream_client_logs, args=(client_proc, i, log_file), daemon=True).start()
        time.sleep(2) # Increased delay to avoid race on server binding

    print(f"\n{'=' * 60}")
    print(f"  ALL SYSTEMS ONLINE")
    print(f"  Bridge:    http://localhost:{BRIDGE_PORT}")
    print(f"  WebSocket: ws://localhost:{BRIDGE_PORT}/ws")
    print(f"  Dashboard: Open http://localhost:5173 (or 5174)")
    print(f"  Logs:      tail -f backend.log")
    print(f"{'=' * 60}")
    print("  Press Ctrl+C to shut down.\n")

    # Keep alive & monitor
    while True:
        # Check if server process died
        if server_proc.poll() is not None:
            print("  [ERROR] Server process exited. Check backend.log")
            break
        time.sleep(2)

if __name__ == "__main__":
    main()
