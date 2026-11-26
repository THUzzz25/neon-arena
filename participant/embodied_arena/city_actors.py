"""Deterministic prescribed motion. Contestant robots remain fully dynamic."""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from .city import beam, geom, ring, numbers

VEHICLES = [
    {"name":"traffic/azure","lane":-1.25,"direction":1,"speed":1.6,"offset":18.},
    {"name":"traffic/violet","lane":1.25,"direction":-1,"speed":1.3,"offset":34.},
    {"name":"traffic/courier","lane":-1.25,"direction":1,"speed":1.6,"offset":43.},
    {"name":"traffic/living_taxi","lane":28.5,"direction":1,"speed":1.2,"offset":21.,"accent":"amber"},
    {"name":"traffic/living_delivery","lane":31.5,"direction":-1,"speed":1.1,"offset":42.,"accent":"mint"},
    {"name":"traffic/living_cab","lane":28.5,"direction":1,"speed":1.2,"offset":51.,"accent":"amber"},
]
PEDESTRIANS=[
    {"name":"citizen/01","kind":"robot","center":(-9.5,4.65),"radius":4.6,"speed":.50,"phase":0.},
    {"name":"citizen/02","kind":"robot","center":(10.,-4.7),"radius":4.4,"speed":.45,"phase":2.},
    {"name":"dog/01","kind":"dog","center":(8.,3.55),"radius":5.3,"speed":.68,"phase":1.},
    {"name":"dog/02","kind":"dog","center":(-10.,-3.55),"radius":4.8,"speed":.60,"phase":3.2},
    {"name":"dog/03","kind":"dog","center":(22.,3.55),"radius":4.5,"speed":.65,"phase":.5},
    {"name":"dog/04","kind":"dog","center":(22.,-3.55),"radius":4.,"speed":.63,"phase":2.4},
    {"name":"dog/05","kind":"dog","center":(-23.,3.55),"radius":4.4,"speed":.60,"phase":1.8},
    {"name":"dog/06","kind":"dog","center":(-23.,-3.55),"radius":4.4,"speed":.66,"phase":4.1},
    {"name":"citizen/03","kind":"robot","center":(8.,24.6),"radius":4.4,"speed":.45,"phase":1.},
    {"name":"dog/07","kind":"dog","center":(-10.,26.2),"radius":5.,"speed":.60,"phase":1.8},
    {"name":"dog/08","kind":"dog","center":(22.,33.7),"radius":4.,"speed":.60,"phase":2.7},
]
DRONES=[]
for band,(count,height,rx,ry,omega) in enumerate(((3,4.2,20.,1.,.055),(2,8.8,26.,1.8,.040),(2,15.5,29.,2.8,.028))):
    for j in range(count):
        DRONES.append({"name":f"drone/{len(DRONES)+1:02d}","band":band,"height":height+j*.12,
                       "rx":rx,"ry":ry,"cy":30. if j==1 else 0.,"omega":omega,"phase":j*2*np.pi/count+band*.6,"scale":1+band*.15})


def yaw_quat(angle):
    return np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])


def multiply(a,b):
    out=np.empty(4)
    mujoco.mju_mulQuat(out,a,b)
    return out


def pedestrian_pose(spec,t):
    phase=t*spec["speed"]/spec["radius"]+spec["phase"]
    x=spec["center"][0]+spec["radius"]*np.sin(phase)
    direction=1 if np.cos(phase)>=0 else -1
    height=.82 if spec["kind"]=="robot" else .62
    return np.array([x,spec["center"][1],height+.018*np.sin(t*7)]),yaw_quat(0 if direction==1 else np.pi)


def vehicle_pose(spec,t):
    x=-36.+(spec["offset"]+t*spec["speed"])%72.
    return np.array([x*spec["direction"],spec["lane"],.65+.035*np.sin(t*2.2+spec["offset"])]),yaw_quat(0 if spec["direction"]==1 else np.pi)


def drone_pose(spec,t):
    angle=t*spec["omega"]+spec["phase"]
    pos=np.array([spec["rx"]*np.cos(angle),spec.get("cy",0.)+spec["ry"]*np.sin(angle),spec["height"]+.18*np.sin(t*.7+spec["phase"])])
    yaw=np.arctan2(spec["ry"]*np.cos(angle),-spec["rx"]*np.sin(angle))
    return pos,yaw_quat(yaw)


def mocap(world,name,pos,quat=(1,0,0,0)):
    return ET.SubElement(world,"body",name=name,mocap="true",pos=numbers(pos),quat=numbers(quat))


def add_actors(world,traffic_enabled=True):
    if traffic_enabled:
        for i,spec in enumerate(VEHICLES):
            name=spec["name"]; position,quat=vehicle_pose(spec,0)
            body=mocap(world,name,position,quat)
            accent=spec.get("accent","cyan" if i%2==0 else "purple")
            geom(body,name+"/hull","box",(0,0,0),(1.25,.58,.18),"metal",mass="180")
            geom(body,name+"/nose","ellipsoid",(.55,0,.08),(.9,.60,.27),"panel",False)
            geom(body,name+"/cockpit","ellipsoid",(-.12,0,.31),(.76,.47,.30),"glass",False)
            for side in (-1,1):
                beam(body,name+f"/rail_{side}",(-1.15,side*.60,.02),(1.1,side*.60,.02),.025,accent)
                beam(body,name+f"/headlight_{side}",(1.27,side*.48,.10),(1.34,side*.25,.10),.045,"cyan")
                beam(body,name+f"/tail_{side}",(-1.27,side*.48,.08),(-1.27,side*.22,.08),.04,"pink")
                for x in (-.85,.85):
                    geom(body,name+f"/thruster_{side}_{x}","cylinder",(x,side*.42,-.23),(.18,.10),"dark",False)
                    geom(body,name+f"/exhaust_{side}_{x}","ellipsoid",(x,side*.42,-.36),(.13,.13,.17),accent,False)
            if "living_" in name:
                geom(body,name+"/cab_sign",pos=(0,0,.72),size=(.23,.13,.07),material=accent,solid=False)
    for spec in PEDESTRIANS:
        name=spec["name"]; position,quat=pedestrian_pose(spec,0)
        body=mocap(world,name,position,quat)
        if spec["kind"]=="robot":
            geom(body,name+"/collision","capsule",size=(.20,),fromto=(0,0,-.55,0,0,.45),material="metal")
            geom(body,name+"/shell","box",(0,0,.25),(.20,.145,.24),"panel",False)
            geom(body,name+"/head","ellipsoid",(0,0,.65),(.16,.13,.14),"metal",False)
            geom(body,name+"/visor","box",(.145,0,.65),(.018,.105,.045),"cyan",False)
            beam(body,name+"/chest",(.205,-.11,.35),(.205,.11,.35),.022,"purple")
            for side in (-1,1):
                limb=mocap(world,name+f"/leg_{side}",position,quat)
                beam(limb,name+f"/shin_{side}",(0,0,0),(.04,0,-.46),.055,"metal")
                geom(limb,name+f"/foot_{side}","box",(.10,0,-.47),(.14,.075,.04),"panel",False)
                limb=mocap(world,name+f"/arm_{side}",position,quat)
                beam(limb,name+f"/upper_arm_{side}",(0,0,0),(.03,0,-.38),.052,"metal")
        else:
            geom(body,name+"/collision","box",(0,0,0),(.38,.18,.16),"metal")
            geom(body,name+"/head","box",(.44,0,.08),(.18,.14,.13),"panel",False)
            geom(body,name+"/visor","box",(.626,0,.10),(.012,.10,.035),"cyan",False)
            beam(body,name+"/backlight",(-.3,0,.17),(.25,0,.17),.023,"purple")
            for x in (-.22,0,.22):
                geom(body,name+f"/spine_plate_{x}","box",(x,0,.18),(.085,.16,.025),"panel",False)
            for side in (-1,1):
                beam(body,name+f"/ear_{side}",(.43,side*.11,.20),(.40,side*.14,.35),.025,"metal")
                geom(body,name+f"/sensor_{side}","sphere",(.636,side*.07,.13),(.026,),"amber",False)
            for front in (-1,1):
                for side in (-1,1):
                    limb=mocap(world,name+f"/leg_{front}_{side}",position,quat)
                    beam(limb,name+f"/thigh_{front}_{side}",(0,0,0),(-.07,0,-.22),.035,"metal")
                    beam(limb,name+f"/calf_{front}_{side}",(-.07,0,-.22),(.07,0,-.43),.029,"panel")
                    geom(limb,name+f"/knee_{front}_{side}","cylinder",(-.07,0,-.22),(.052,.035),"metal",False,euler=(np.pi/2,0,0))
    for spec in DRONES:
        name=spec["name"];s=spec["scale"];pos,quat=drone_pose(spec,0)
        body=mocap(world,name,pos,quat)
        accent=("cyan","amber","pink")[spec["band"]]
        geom(body,name+"/body","ellipsoid",size=(.32*s,.26*s,.13*s),material="metal",solid=False)
        geom(body,name+"/eye","sphere",(.25*s,0,-.05*s),(.065*s,),accent,False)
        geom(body,name+"/cargo","box",(-.04*s,0,-.18*s),(.18*s,.15*s,.07*s),"panel",False)
        for sx in (-1,1):
            for sy in (-1,1):
                beam(body,name+f"/arm_{sx}_{sy}",(0,0,0),(sx*.52*s,sy*.52*s,0),.04*s,"metal")
                ring(body,name+f"/rotor_{sx}_{sy}",(sx*.52*s,sy*.52*s,.02*s),.23*s,accent,12,.016*s)
                rotor=mocap(world,name+f"/blade_{sx}_{sy}",pos,quat)
                beam(rotor,name+f"/blade_geom_{sx}_{sy}",(-.19*s,0,0),(.19*s,0,0),.018*s,"metal")
    train=mocap(world,"transit/skytrain",(-26,53,10.8))
    for i in (-1,0,1):
        geom(train,f"skytrain_body_{i}",pos=(i*3,0,0),size=(1.4,.64,.52),material="metal",solid=False)
        for side in (-1,1):
            beam(train,f"skytrain_line_{i}_{side}",(i*3-1.3,side*.66,-.32),(i*3+1.3,side*.66,-.32),.025,"amber")
            for j in range(4):
                geom(train,f"skytrain_window_{i}_{side}_{j}",pos=(i*3-.9+j*.6,side*.65,.12),size=(.22,.015,.23),material="cyan",solid=False)


class CityMotion:
    def __init__(self,model):
        self.model=model
        self._background_frame=None
        self.ids={model.body(i).name:int(mid) for i,mid in enumerate(model.body_mocapid) if mid>=0}
        self.vehicle_geoms={}
        for gid in range(model.ngeom):
            name=model.geom(gid).name or ""
            if name.startswith("traffic/"):
                self.vehicle_geoms[gid]="/".join(name.split("/")[:2])

    def set_pose(self,data,name,pos,quat):
        mid=self.ids.get(name)
        if mid is not None:
            data.mocap_pos[mid]=pos;data.mocap_quat[mid]=quat

    def update(self,data,t):
        for spec in VEHICLES:
            pos,quat=vehicle_pose(spec,t);self.set_pose(data,spec["name"],pos,quat)
        # Exact vehicle motion stays at physics rate; decorative animation is
        # evaluated on deterministic 50 Hz frames rather than repeating work.
        frame=int(np.floor((t+1e-9)/.02))
        if frame==self._background_frame:return
        self._background_frame=frame;t=frame*.02
        for spec in PEDESTRIANS:
            pos,quat=pedestrian_pose(spec,t);name=spec["name"]
            self.set_pose(data,name,pos,quat)
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,quat);rotation=rotation.reshape(3,3)
            gait=t*6.+spec["phase"]
            if spec["kind"]=="robot":
                for side in (-1,1):
                    angle=.34*np.sin(gait+(0 if side==1 else np.pi))
                    pitch=np.array([np.cos(angle/2),0,np.sin(angle/2),0])
                    self.set_pose(data,name+f"/leg_{side}",pos+rotation@np.array([0,side*.11,-.23]),multiply(quat,pitch))
                    self.set_pose(data,name+f"/arm_{side}",pos+rotation@np.array([0,side*.25,.39]),multiply(quat,pitch*np.array([1,1,-1,1])))
            else:
                for front in (-1,1):
                    for side in (-1,1):
                        angle=.38*np.sin(gait+(0 if side*front==1 else np.pi))
                        pitch=np.array([np.cos(angle/2),0,np.sin(angle/2),0])
                        self.set_pose(data,name+f"/leg_{front}_{side}",pos+rotation@np.array([front*.26,side*.17,-.05]),multiply(quat,pitch))
        for spec in DRONES:
            pos,quat=drone_pose(spec,t);s=spec["scale"];name=spec["name"]
            self.set_pose(data,name,pos,quat)
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,quat);rotation=rotation.reshape(3,3)
            for sx in (-1,1):
                for sy in (-1,1):
                    self.set_pose(data,name+f"/blade_{sx}_{sy}",pos+rotation@np.array([sx*.52*s,sy*.52*s,.025*s]),multiply(quat,yaw_quat(t*30)))
        self.set_pose(data,"transit/skytrain",np.array([-36+(10+t*2.4)%72,53,10.8]),np.array([1.,0,0,0]))

    @staticmethod
    def vehicle_speed(name):
        return next(v["speed"] for v in VEHICLES if v["name"]==name)
