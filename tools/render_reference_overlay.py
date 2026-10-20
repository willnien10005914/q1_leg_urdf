#!/usr/bin/env python3
"""Rebuild only the video overlay from saved poses; leave trained motion unchanged."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from extract_sidestep import EDGES

root=Path(__file__).resolve().parents[1]/'docs/reference'
data=json.loads((root/'sidestep_landmarks.json').read_text())
times=np.asarray(data['time']);poses=np.asarray(data['image'])
capture=cv2.VideoCapture(str(root/'sidestep_0_7.mp4'))
fps=capture.get(cv2.CAP_PROP_FPS)
if not capture.isOpened() or fps<=0:raise SystemExit('Cannot decode reference clip')
width,height=int(capture.get(3)),int(capture.get(4))
temporary=root/'sidestep_overlay_preview_raw.mp4'
writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
if not writer.isOpened():raise SystemExit('Cannot create reference overlay')
index=0
try:
    while index/fps<7:
        ok,frame=capture.read()
        if not ok:break
        t=index/fps;i=int(np.argmin(abs(times-t)))
        if abs(times[i]-t)<=.1:
            for a,b in EDGES:
                cv2.line(frame,tuple((poses[i,a,:2]*[width,height]).astype(int)),tuple((poses[i,b,:2]*[width,height]).astype(int)),(0,255,130),2)
        cv2.putText(frame,f'{t:.2f}s saved MediaPipe pose',(20,height-24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        writer.write(frame);index+=1
finally:
    capture.release();writer.release()
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'sidestep_pose_overlay.mp4')],check=True)
temporary.unlink()
print(f'Rebuilt {index} overlay frames; joint reference and foot targets unchanged.')
