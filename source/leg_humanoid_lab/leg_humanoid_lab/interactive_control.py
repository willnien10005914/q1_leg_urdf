"""Finite keyboard/gesture controller; never teleports or resets physics."""
import math

MODES={'stand':0,'walk':1,'sidestep_left':2,'sidestep_right':3,'recover':4}
LIMITS=(.22,.18,.6)

class KeyboardController:
    def __init__(self,sidestep_duration=1.6,timeout=.5):
        self.duration=float(sidestep_duration)
        self.timeout=float(timeout)
        self.wanted=[0.,0.,0.];self.command=[0.,0.,0.]
        self.last_input=-math.inf;self.sequence_active=False
        self.stage='stand';self.clock=0.;self.gait_phase=0.

    def velocity(self,vx,vy,wz,now):
        values=[float(vx),float(vy),float(wz)]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite velocity')
        if self.sequence_active or self.stage in ('prepare','recover'):return False
        values=[max(-limit,min(limit,v)) for v,limit in zip(values,LIMITS)]
        length=sum(abs(v)/limit for v,limit in zip(values,LIMITS))
        if length>1:values=[v/length for v in values]
        self.wanted=values;self.last_input=now
        return True

    def sequence(self,now):
        if self.sequence_active:return False
        self.wanted=[0.,0.,0.];self.last_input=now
        self.sequence_active=True;self.stage='prepare';self.clock=0.
        return True

    def stop(self,now):
        self.wanted=[0.,0.,0.];self.last_input=now
        self.sequence_active=False;self.stage='recover';self.clock=0.

    def tick(self,dt,now):
        dt=max(0.,min(float(dt),.1))
        self.clock+=dt
        if not self.sequence_active and now-self.last_input>self.timeout:
            self.wanted=[0.,0.,0.]
        alpha=1-math.exp(-dt/.18)
        self.command=[a+alpha*(b-a) for a,b in zip(self.command,self.wanted)]
        if self.stage=='prepare' and self.clock>=.5:
            self.stage='sidestep_left';self.clock=0.
        elif self.stage=='sidestep_left' and self.clock>=self.duration:
            self.stage='sidestep_right';self.clock=0.
        elif self.stage=='sidestep_right' and self.clock>=self.duration:
            self.stage='recover';self.clock=0.;self.sequence_active=False
        elif self.stage=='recover' and self.clock>=1.:
            self.stage='stand';self.clock=0.
        if self.stage in ('sidestep_left','sidestep_right'):
            mode=self.stage;phase=min(1.,self.clock/self.duration)
        elif self.stage in ('prepare','recover'):
            mode='recover';phase=0.
        else:
            mode='walk' if math.hypot(*self.command[:2])>.008 or abs(self.command[2])>.02 else 'stand'
            phase=0.
        return {'mode':mode,'command':list(self.command),'phase':phase,'sequence_active':self.sequence_active}
