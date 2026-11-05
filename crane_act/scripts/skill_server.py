#!/usr/bin/env python3
"""Resident skill server: keeps every ACT policy of skills.json loaded and runs skill requests.

Removes the ~25 s start-up (pixi, torch/lerobot import, model load, CUDA warm-up) of each
run_skill.sh call. Protocol: one JSON line per TCP connection on 127.0.0.1:9877 (inside the container)
  request : {"skill": "wave", "seconds": 8.0}      (seconds optional -> default_seconds)
  response: {"ok": true, "skill": "wave", "seconds": 8.0, "home_reached": true,
             "eval": "success=True ...", "log": "~/logs/runs/..."}
Start: bash /workspace/a_crane/scripts/skill_server.sh start   (SIM=demo by default)
"""
import json
import os
import socketserver
import sys
import threading
import time

import rclpy
import torch
from rclpy.executors import SingleThreadedExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_runner as sr  # noqa: E402
from eval_traj import evaluate  # noqa: E402
from lerobot.common.policies.act.modeling_act import ACTPolicy  # noqa: E402

SKILLS_JSON = '/workspace/a_crane/skills.json'
PORT = int(os.environ.get('SKILL_SERVER_PORT', '9877'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_all():
    cfg = json.load(open(SKILLS_JSON, encoding='utf-8'))
    pols = {}
    for name, c in cfg.items():
        p = ACTPolicy.from_pretrained(c['model_path']).eval().to(DEVICE)
        meta = os.path.join(c['model_path'], 'a_crane_meta.json')
        m = json.load(open(meta)) if os.path.exists(meta) else {}
        mode = m.get('env_mode', 'state')
        chunk_exec = int(m.get('chunk_exec', 1))
        with torch.no_grad():  # warm-up
            for _ in range(3):
                p.reset()
                p.select_action({'observation.state': torch.from_numpy(sr.HOME).unsqueeze(0).to(DEVICE),
                                 'observation.environment_state': torch.zeros(1, 5).to(DEVICE)})
        pols[name] = (p, mode, float(c.get('default_seconds', 8.0)), chunk_exec)
        print(f'loaded {name} ({mode}, chunk_exec={chunk_exec})', flush=True)
    return pols


def main():
    rclpy.init()
    node = sr.Runner()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    pols = load_all()
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                req = json.loads(self.rfile.readline().decode())
                name = req['skill']
                if name == '__list__':
                    self.wfile.write((json.dumps({'ok': True, 'skills': list(pols)}) + '\n').encode())
                    return
                if name not in pols:
                    raise KeyError(f'unknown skill {name}')
                pol, mode, dflt, chunk_exec = pols[name]
                sec = float(req.get('seconds') or dflt)
                with lock:
                    t_w = time.time()
                    while node.js is None and time.time() - t_w < 5:
                        time.sleep(0.1)
                    if node.js is None:
                        raise RuntimeError('no /joint_states (simulation not running?)')
                    sr.go_home(node)
                    log = os.path.expanduser(f'~/logs/runs/{name}_{time.strftime("%Y%m%d_%H%M%S")}.csv')
                    marker = req.get('start_marker')
                    if marker:
                        open(marker, 'w').close()
                        time.sleep(float(req.get('start_delay', 0)))
                    ok = sr.run_once(node, pol, mode, sec, DEVICE, log, chunk_exec)
                ev = evaluate(name, log)
                resp = {'ok': True, 'skill': name, 'seconds': sec, 'home_reached': bool(ok),
                        'wall': round(getattr(sr.run_once, 'last_wall', 0.0), 2),
                        'eval': f'success={ev[0]} motion={ev[1]} home_err={ev[2]:.3f} {ev[3]}', 'log': log}
            except Exception as e:  # report errors to the client
                resp = {'ok': False, 'error': repr(e)}
            self.wfile.write((json.dumps(resp, ensure_ascii=False) + '\n').encode())

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(('127.0.0.1', PORT), Handler) as srv:
        print(f'skill server ready on 127.0.0.1:{PORT} skills={list(pols)}', flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
