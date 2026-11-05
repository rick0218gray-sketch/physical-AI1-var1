#!/usr/bin/env python3
"""Run a trained ACT skill policy on CRANE+ V2 (Gazebo) for N seconds, then return to HOME.

Extends pai_training/scripts/infer.py:
  * fixed duration, then HOME (upright, gripper open) is commanded until reached
  * supports env-mode 'phase' (environment_state = [t, dq1..dq4]) and 'state' (lecture)
  * optional CSV log of (t, state, action) for evaluation
Runs in the pai_training pixi env with ROS 2 sourced:
  cd ~/pai_training && pixi run python /workspace/a_crane/scripts/skill_runner.py --model-path ... --seconds 8
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'
JS_ORDER = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
HOME = np.array([0.0, 0.0, 0.0, 0.0, -0.5236], dtype=np.float32)  # j1 j2 j4 j3 hand
DT = 0.05


class Runner(Node):
    def __init__(self):
        super().__init__('crane_plus_skill_runner',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.cb, 10)
        self.js = None

    def cb(self, msg):
        d = dict(zip(msg.name, msg.position))
        self.js = np.array([d.get(n, 0.0) for n in JS_ORDER], dtype=np.float32)

    def publish(self, a):
        jt = JointTrajectory()
        jt.joint_names = ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in a[:4]]
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        jg = JointTrajectory()
        jg.joint_names = [GRIPPER_JOINT_NAME]
        pg = JointTrajectoryPoint()
        pg.positions = [float(a[4])]
        pg.time_from_start.nanosec = int(0.1 * 1e9)
        jg.points.append(pg)
        self.gripper_pub.publish(jg)


def spin_for(node, sec):
    # callbacks (/joint_states, /clock) are processed by a background executor thread;
    # spinning them one at a time from this loop made the sim clock lag behind (/clock is ~1 kHz)
    time.sleep(max(0.0, sec))


def go_home(node, timeout=6.0, tol=0.03, pose=None):
    target = HOME if pose is None else pose
    t0 = time.time()
    while time.time() - t0 < timeout:
        node.publish(target)
        spin_for(node, DT)
        if node.js is not None and np.max(np.abs(node.js - target)) < tol:
            return True
    return False


def run_once(node, policy, mode, seconds, device, log_path, chunk_exec=1):
    # chunk_exec > 1: execute that many consecutive actions of each predicted chunk open-loop.
    # The chunk elements are 0.1 s apart, so a new element is taken every 2 control ticks (0.05 s).
    policy.config.n_action_steps = chunk_exec
    policy.reset()
    a = None
    hist = deque(maxlen=3)
    rows = []
    sim_now = lambda: node.get_clock().now().nanoseconds * 1e-9
    t0 = time.time()
    s0 = sim_now()
    i = 0
    while True:
        t = round(sim_now() - s0, 3)  # actual elapsed sim time (robust to a slow inference step)
        if t >= seconds:
            break
        s = node.js.copy()
        hist.append(s)
        if mode == 'phase':
            env = np.array([t, 0, 0, 0, 0], dtype=np.float32)
        elif mode == 'phasedq':
            prev = hist[0]
            env = np.concatenate([[t], s[:4] - prev[:4]]).astype(np.float32)
        else:
            env = s
        obs = {'observation.state': torch.from_numpy(s).unsqueeze(0).to(device),
               'observation.environment_state': torch.from_numpy(env).unsqueeze(0).to(device)}
        if chunk_exec == 1 or i % 2 == 0 or a is None:
            with torch.no_grad():
                a = policy.select_action(obs).cpu().numpy().flatten()
        node.publish(a)
        rows.append([round(t, 3)] + s.tolist() + a.tolist())
        i += 1
        # 20 Hz on the SIMULATION clock (the training data was resampled on sim time)
        while sim_now() < s0 + i * DT:
            time.sleep(0.002)

    ok = go_home(node)
    # wall-clock duration of the visible motion (policy run + return to upright); D's commander uses it
    run_once.last_wall = time.time() - t0
    print(f'done: {seconds}s mode={mode} home_reached={ok} steps={i} wall={time.time() - t0:.1f}s sim={sim_now() - s0:.1f}s', flush=True)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t'] + [f's_{n}' for n in ('j1', 'j2', 'j4', 'j3', 'g')] +
                       [f'a_{n}' for n in ('j1', 'j2', 'j4', 'j3', 'g')])
            w.writerows(rows)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--seconds', type=float, default=8.0)
    ap.add_argument('--env-mode', default=None, help='phase|state (default: a_crane_meta.json in model dir)')
    ap.add_argument('--log', default=None, help='CSV path for (t, state, action)')
    ap.add_argument('--no-home-first', action='store_true')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--start-pose', default=None, help='comma separated j1,j2,j4,j3,hand to start from')
    ap.add_argument('--chunk-exec', type=int, default=None,
                    help='open-loop actions per chunk (default: a_crane_meta.json chunk_exec or 1)')
    ap.add_argument('--eval-skill', default=None, help='evaluate each trial with eval_traj.py criteria')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    mode = args.env_mode
    meta_p = os.path.join(args.model_path, 'a_crane_meta.json')
    if mode is None:
        mode = json.load(open(meta_p)).get('env_mode', 'state') if os.path.exists(meta_p) else 'state'

    if args.chunk_exec is None:
        args.chunk_exec = json.load(open(meta_p)).get('chunk_exec', 1) if os.path.exists(meta_p) else 1
    policy = ACTPolicy.from_pretrained(args.model_path)
    policy.eval().to(args.device)
    policy.reset()

    rclpy.init()
    node = Runner()
    import threading
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    t_wait = time.time()
    while node.js is None and time.time() - t_wait < 10:
        spin_for(node, 0.1)
    if node.js is None:
        print('no /joint_states; is the simulation running?')
        return 2
    start_pose = None
    if args.start_pose:  # robustness test: start from a perturbed posture (j1 j2 j4 j3 hand order)
        start_pose = np.array([float(x) for x in args.start_pose.split(',')], dtype=np.float32)
        go_home(node, pose=start_pose)
        spin_for(node, 0.3)
    elif not args.no_home_first:
        go_home(node)
        spin_for(node, 0.3)

    # warm-up (CUDA kernels, cudnn autotune) so that the first real step is not ~1 s late
    with torch.no_grad():
        for _ in range(3):
            policy.reset()
            policy.select_action({'observation.state': torch.from_numpy(HOME).unsqueeze(0).to(args.device),
                                  'observation.environment_state': torch.zeros(1, 5).to(args.device)})
    if os.environ.get('START_MARKER'):  # lets demo_record.sh start the screen capture now
        time.sleep(float(os.environ.get('START_DELAY', '0')))
        open(os.environ['START_MARKER'], 'w').close()
        time.sleep(float(os.environ.get('START_DELAY', '0')))
    for k in range(args.trials):
        log_path = args.log
        if args.log and args.trials > 1:
            root, ext = os.path.splitext(args.log)
            log_path = f'{root}_{k:02d}{ext}'
        run_once(node, policy, mode, args.seconds, args.device, log_path, args.chunk_exec)
        if args.eval_skill and log_path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from eval_traj import evaluate
            ok, motion, err, detail = evaluate(args.eval_skill, log_path)
            print(f'trial {k}: success={ok} motion={motion} home_err={err:.3f} {detail}', flush=True)
        if k + 1 < args.trials:
            spin_for(node, 0.5)
            if args.start_pose:
                go_home(node, pose=start_pose)
                spin_for(node, 0.3)
    ex.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
