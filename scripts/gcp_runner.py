import subprocess
import requests
import json
import websocket
import time
import sys

# Get GCP credentials and notebook proxy URL
def get_gcloud_token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()

# Hardcoded details from our discovery
PROXY_HOST = "b646fe1694b2c75-dot-europe-west1.notebooks.googleusercontent.com"
BASE_URL = f"https://{PROXY_HOST}"
WS_URL = f"wss://{PROXY_HOST}"

def run_command_remote(command_str):
    token = get_gcloud_token()
    
    # Create session
    session = requests.Session()
    
    # 1. Authenticate via the GCP Notebooks sign-in endpoint
    # This endpoint will validate our Bearer token and redirect us to the proxy URL
    # with a DATALAB_TUNNEL_TOKEN query parameter, which sets the true session cookies.
    continue_url = f"{BASE_URL}/lab"
    signin_url = f"https://europe-west1.notebooks.cloud.google.com/_signin?continue={requests.utils.quote(continue_url)}&endpoint=b646fe1694b2c75"
    
    print("Performing GCP sign-in flow...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Follow redirects to complete sign-in and populate cookies
    res = session.get(signin_url, headers=headers, allow_redirects=True)
    if res.status_code != 200:
        print(f"Sign-in failed with status {res.status_code}")
        return
        
    print("Sign-in successful!")
    
    # Extract XSRF token
    xsrf_token = session.cookies.get("_xsrf")
    if not xsrf_token:
        # Check if we can find it in the page content or other cookies
        for cookie in session.cookies:
            if cookie.name == "_xsrf":
                xsrf_token = cookie.value
                break
                
    if not xsrf_token:
        print("Warning: _xsrf token not found, trying to load /lab to get it...")
        lab_res = session.get(f"{BASE_URL}/lab")
        xsrf_token = session.cookies.get("_xsrf")
        
    if not xsrf_token:
        print("Warning: Could not find _xsrf token in cookies, using a dummy one")
        xsrf_token = "dummy"

    # 2. Create a new terminal session
    post_headers = {
        "Authorization": f"Bearer {token}",
        "X-XSRFToken": xsrf_token
    }
    
    term_res = session.post(f"{BASE_URL}/api/terminals", headers=post_headers)
    if term_res.status_code != 200:
        print(f"Error creating terminal: {term_res.status_code} - {term_res.text}")
        return
        
    term_data = term_res.json()
    terminal_name = term_data["name"]
    print(f"Created remote terminal: {terminal_name}")
    
    # 3. Connect to terminal via WebSocket using ONLY session cookies (no Authorization to avoid 500)
    cookie_str = "; ".join([f"{c.name}={c.value}" for c in session.cookies])
    print(f"Debug - Cookie String: {cookie_str}")
    
    ws_headers = [
        f"Cookie: {cookie_str}",
        f"Origin: {BASE_URL}"
    ]
    
    ws_endpoint = f"{WS_URL}/terminals/websocket/{terminal_name}"
    
    output_buffer = []
    command_sent = False
    finished = False
    
    def on_message(ws, message):
        nonlocal command_sent, finished
        msg = json.loads(message)
        msg_type, content = msg[0], msg[1]
        
        if msg_type == "stdout":
            sys.stdout.write(content)
            sys.stdout.flush()
            output_buffer.append(content)
            
            # Since terminal is interactive, we look for prompt characters or completion cues
            # to know when to exit if running a single command.
            # But wait! A simpler way to execute a command and close is to chain it with an exit command!
            # e.g., "echo '__DONE__' && exit\n"
            if "__DONE__" in content:
                finished = True
                ws.close()
                
        elif msg_type == "disconnect":
            finished = True
            ws.close()

    def on_open(ws):
        nonlocal command_sent
        # Send our command followed by an echo to signal completion and exit the terminal shell
        full_cmd = f"{command_str} && echo '__DONE__' && exit\n"
        ws.send(json.dumps(["stdin", full_cmd]))
        command_sent = True

    def on_error(ws, error):
        print(f"WS Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        pass

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        ws_endpoint,
        header=ws_headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    ws.run_forever(origin=BASE_URL)

if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ls -la"
    print(f"Running command remotely on VM: {cmd}")
    run_command_remote(cmd)
