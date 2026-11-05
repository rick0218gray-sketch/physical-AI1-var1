#!/usr/bin/env python3
"""Record the "natural language -> LLM -> ACT" demo clip (runs on the WSL host side).

1. starts an ffmpeg x11grab recording of the Gazebo view inside the pai_ros2 container
2. runs d_llm/commander.py (unchanged, read-only use) for each command, time-stamping its output
3. writes subtitles (command / LLM decision / running skill) as SRT, burns them in, and cuts out
   the waiting time -> a_crane/videos/demo_nl_commander.mp4 (+ the raw recording)

usage: python3 record_nl_demo.py "左を指差してから、手を振って" "コップを取ってきて" ...
"""
import json
import os
import re
import subprocess
import sys
import time

import glob
import shutil
A_HOST = os.environ.get('A_HOST', os.path.expanduser('~/pai_final/a_crane'))
D_LLM = os.environ.get('D_LLM', os.path.expanduser('~/pai_final/d_llm'))
DOCKER = (os.environ.get('DOCKER') or shutil.which('docker') or shutil.which('docker.exe')
          or next(iter(glob.glob('/mnt/c/Users/*/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe')), 'docker'))
C = 'pai_ros2'
VID = '/workspace/a_crane/videos'
TS = time.strftime('%H%M%S')
RAW = f'{VID}/demo_nl_raw_{TS}.mp4'
SKILL_JA = {'bow': 'お辞儀', 'nod': 'うなずき', 'wave': '手を振る', 'point_left': '左を指差す',
            'point_right': '右を指差す', 'grip': 'つかむ'}


def dexec(cmd, detach=False):
    args = [DOCKER, 'exec'] + (['-d'] if detach else []) + ['-u', 'ubuntu', C, 'bash', '-c', cmd]
    return subprocess.run(args, capture_output=True, text=True)


def srt_time(t):
    # round to whole milliseconds first: rounding only the fractional part turned e.g. 1.9996 s
    # into "00:00:01,1000" (and "00:00:01.100" after the ASS truncation below)
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def main():
    commands = []
    for a in sys.argv[1:]:  # "@file" = one command per line (UTF-8; avoids argv encoding issues via wsl.exe)
        commands += [l.strip() for l in open(a[1:], encoding='utf-8') if l.strip()] if a.startswith('@') else [a]
    dexec('CAM_VIEW=top bash /workspace/a_crane/scripts/gazebo_window.sh')
    # recording (raw, long; stopped with SIGINT at the end)
    dexec(f'ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -video_size 1184x928 -framerate 20 '
          f'-i :0.0+0,112 -t 600 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p {RAW}', detach=True)
    t_rec = time.time() + 0.4  # ffmpeg start-up latency (approx.)
    time.sleep(2.0)
    cues = []      # (start, end, text, layer) layer 0 = top line (command), 1 = bottom (status)
    keep = []      # segments of the raw video to keep
    for cmd in commands:
        t0 = time.time() - t_rec
        cues_cmd = []
        p = subprocess.Popen(['python3', '-u', 'commander.py', '--container', C, '--container-env', 'AUTO_START_SIM=0',
                              '--skills', f'{A_HOST}/skills.json', '--log', f'{A_HOST}/results/nl_demo_commander.jsonl',
                              cmd], cwd=D_LLM, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t_decide, decision, message, cur = None, None, '', None
        steps = []
        for line in p.stdout:
            now = time.time() - t_rec
            sys.stdout.write(f'[{now:7.2f}] {line}')
            m = re.match(r'判定: (\w+)', line.strip())
            if m:
                t_decide, decision = now, m.group(1)
            m = re.match(r'応答: (.*)', line.strip())
            if m:
                message = m.group(1)
            m = re.match(r'\[(\d+)/(\d+)\] .*run_skill\.sh\S* (\w+) ([0-9.]+)', line.strip())
            if m:
                cur = {'skill': m.group(3), 'i': int(m.group(1)), 'n': int(m.group(2)), 't_cmd': now}
            m = re.search(r'done: .*wall=([0-9.]+)s', line)
            if m and cur:
                cur['t_end'] = now
                cur['t_start'] = now - float(m.group(1))
            m = re.search(r'success=(True|False)', line)
            if m and cur:
                cur['ok'] = m.group(1) == 'True'
                steps.append(cur)
                cur = None
        p.wait()
        t1 = time.time() - t_rec
        t_end_cmd = max(t1 + 1.0, (t_decide or t1) + 3.5)
        cues.append((t0, t_end_cmd, f'指令「{cmd}」', 0))
        if t_decide is not None:
            label = {'ok': '実行', 'reject': '拒否', 'clarify': '確認'}.get(decision, decision)
            plan = ' → '.join(SKILL_JA.get(s['skill'], s['skill']) for s in steps) if steps else ''
            first = steps[0]['t_start'] if steps else t_decide + 3.5  # reject/clarify: keep the answer readable
            cues.append((t_decide, first, f'LLM の判定: {label}　{message}', 1))
            # keep: the command appears (1.5 s), then from the decision on (LLM latency cut out)
            keep.append((t0, t0 + 1.5))
            keep.append((t_decide - 0.3, max(t1 + 1.0, t_decide + 3.5)))
            for s in steps:
                txt = f"実行中 [{s['i']}/{s['n']}] {SKILL_JA.get(s['skill'], s['skill'])}（{s['skill']}, ACT）"
                cues.append((s['t_start'], s['t_end'], txt + ('' if s.get('ok', True) else ' ※基準未達'), 1))
                # cut the gap between the decision and the first motion (docker exec start-up)
            if steps:
                keep[-1] = (t_decide - 0.3, t_decide + 2.5)
                prev_end = None
                for s in steps:
                    keep.append((s['t_start'] - 0.3, s['t_end'] + 0.5))
        else:
            keep.append((t0, t1 + 1.0))
        time.sleep(4.0)  # the decision of reject/clarify stays on screen for ~3.5 s
    time.sleep(1.0)
    dexec('pkill -INT -f "[f]fmpeg.*x11grab"')
    time.sleep(3)

    # subtitles: ASS so that the command (top) and the status (bottom) have fixed positions
    font = 'Noto Sans CJK JP'
    ass = ['[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1184', 'PlayResY: 928', '',
           '[V4+ Styles]',
           'Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginV',
           f'Style: Top,{font},40,&H00FFFFFF,&H00000000,&H90000000,1,3,2,0,8,30',
           f'Style: Bottom,{font},34,&H0000FFFF,&H00000000,&H90000000,1,3,2,0,2,40', '',
           '[Events]', 'Format: Layer, Start, End, Style, Text']
    to_ass = lambda t: srt_time(t).replace(',', '.')[:-1]
    srt = []
    for k, (a, b, txt, layer) in enumerate(sorted(cues)):
        ass.append(f'Dialogue: 0,{to_ass(a)},{to_ass(b)},{"Top" if layer == 0 else "Bottom"},{txt}')
        srt += [str(k + 1), f'{srt_time(a)} --> {srt_time(b)}', txt, '']
    base = f'{A_HOST}/videos/demo_nl_raw_{TS}'
    open(base + '.ass', 'w', encoding='utf-8').write('\n'.join(ass) + '\n')
    open(base + '.srt', 'w', encoding='utf-8').write('\n'.join(srt))
    json.dump({'keep': keep, 'cues': cues}, open(base + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # burn in, then keep only the selected segments
    keep = sorted(keep)
    merged = []
    for a, b in keep:
        a = max(0.0, a)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    sel = '+'.join(f'between(t,{a:.2f},{b:.2f})' for a, b in merged)
    out = f'{VID}/demo_nl_commander.mp4'
    vf = (f"ass={VID}/demo_nl_raw_{TS}.ass,select='{sel}',setpts=N/FRAME_RATE/TB")
    r = dexec(f'ffmpeg -loglevel error -y -i {RAW} -vf "{vf}" -an -c:v libx264 -preset veryfast -crf 22 '
              f'-pix_fmt yuv420p {out} && ffprobe -v error -show_entries format=duration -of csv=p=0 {out}')
    print('segments:', [[round(a, 1), round(b, 1)] for a, b in merged])
    print('raw:', RAW, '\nout:', out, 'duration:', r.stdout.strip(), r.stderr[-500:])


if __name__ == '__main__':
    main()
