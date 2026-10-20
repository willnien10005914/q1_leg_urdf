#!/usr/bin/env python3
"""Prepare conservative alternating lift targets from MediaPipe movement timing.

These are planned robot foot clearances, not measured human foot heights.
The original joint reference and existing policy inputs remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks,savgol_filter

ROOT=Path(__file__).resolve().parents[1]

def build_targets(landmarks,motion,clearance=.04):
    source_t=np.asarray(landmarks['time'])
    image=np.asarray(landmarks['image'])
    separation=savgol_filter(abs(image[:,27,0]-image[:,28,0]),11,2)
    fps=1/np.median(np.diff(source_t))
    peaks,_=find_peaks(separation,height=.08,prominence=.06,distance=round(fps))
    times=np.arange(391)*.02
    heights=np.zeros((len(times),2))
    steps=[]
    for peak in peaks:
        start=end=int(peak)
        while start>0 and separation[start]>.04:start-=1
        while end<len(source_t)-1 and separation[end]>.04:end+=1
        left,mid,right=source_t[[start,peak,end]]
        if min(mid-left,right-mid)<.16:continue
        root=np.asarray(motion['root']);mt=np.asarray(motion['time'])
        direction=np.interp(mid,mt,root[:,1])-np.interp(left,mt,root[:,1])
        lead=0 if direction>=0 else 1
        for a,b,sides in ((left,mid,(lead,1-lead)),(mid,right,(1-lead,lead))):
            for j,side in enumerate(sides):
                lo=a+(b-a)*j/2;hi=a+(b-a)*(j+1)/2
                mask=(times>=lo)&(times<=hi)
                heights[mask,side]=clearance*np.sin(np.pi*(times[mask]-lo)/(hi-lo))**2
                steps.append({'side':('left','right')[side],'start':float(lo),'end':float(hi),'peak_clearance_m':clearance})
    if len(steps)!=16:raise ValueError(f'Expected four source movement bursts / sixteen planned foot swings, got {len(steps)}')
    return {'schema':1,'method':'MediaPipe ankle separation bursts; conservative sequential left/right robot swings',
            'limitations':['Clearance is a planned robot target, not calibrated human foot height.','Sequential support adapts source timing; it does not claim exact reconstruction of jumps.','Joint reference is unchanged; improved stepping requires PPO fine tuning and physical evaluation.'],
            'time':times.tolist(),'height_m':heights.tolist(),'steps':steps,'peak_clearance_m':clearance}

def main():
    source=ROOT/'docs/reference/sidestep_landmarks.json'
    motion=ROOT/'docs/reference/sidestep_motion.json'
    data=build_targets(json.loads(source.read_text()),json.loads(motion.read_text()))
    data.update(landmarks_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),robot_urdf_sha256=json.loads(motion.read_text())['robot_urdf_sha256'])
    out=ROOT/'docs/reference/sidestep_foot_targets.json'
    out.write_text(json.dumps(data,indent=2)+'\n')
    print(f'{out}: {len(data["steps"])} alternating foot swings, target {data["peak_clearance_m"]*100:.0f} cm')

if __name__=='__main__':main()
