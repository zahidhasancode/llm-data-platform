"""Send steady requests during a rolling update and count failures."""

import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:30080/v1/sql"
BODY = json.dumps({"question": "How many rows?", "context": "CREATE TABLE t (a INT);"}).encode()
results: list[dict] = []
stop = threading.Event()


def worker() -> None:
    while not stop.is_set():
        start = time.time()
        request = urllib.request.Request(URL, BODY, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                version = json.load(response)["model_version"]
            results.append({"ok": True, "version": version, "seconds": time.time() - start})
        except Exception as error:
            results.append({"ok": False, "error": repr(error)[:120], "seconds": time.time() - start})
        time.sleep(0.5)


threads = [threading.Thread(target=worker) for _ in range(2)]
for t in threads:
    t.start()
time.sleep(int(sys.argv[1]))
stop.set()
for t in threads:
    t.join()
latencies = sorted(r["seconds"] for r in results if r["ok"])
print(json.dumps({
    "requests": len(results),
    "failed": sum(not r["ok"] for r in results),
    "versions_seen": sorted({r["version"] for r in results if r["ok"]}),
    "median_seconds": round(latencies[len(latencies) // 2], 2) if latencies else None,
    "errors": [r["error"] for r in results if not r["ok"]][:3],
}))
