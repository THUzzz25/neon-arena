"""Street trees, usable-looking seating, signs and shared interior finishes."""
import xml.etree.ElementTree as ET
import numpy as np

from .city import geom,beam,numbers,STREETS
from .city_meshes import Mesh
from .city_life import material,tree
from .city_shops import panel
from .city_room_art import FINISHES

ROW_TREE_X=(-26.,-18.5,-13.8,-6.6,1.,10.8,17.7,25.5)
ROW_BENCH_X=(-28.2,-20.3,-8.6,5.,13.,19.5,28.2)


def shared_finishes(assets):
    colors={"oak":((.48,.30,.16,1),.07),"walnut":((.20,.12,.09,1),.04),
        "ivory":((.72,.67,.55,1),.13),"linen":((.55,.45,.34,1),.08),
        "upholstery":((.12,.18,.25,1),.07),"steel":((.30,.34,.38,1),.04),
        "brass":((.50,.35,.14,1),.04),"teal_paint":((.17,.40,.41,1),.09),
        "soil":((.09,.065,.045,1),.03)}
    for name,(rgba,emission) in colors.items():material(assets,name,rgba,emission,.65 if name in ("brass","steel") else .22)
    for kind in FINISHES:
        ET.SubElement(assets,"texture",name="city_floor_"+kind,type="2d",file=f"city_floor_{kind}.png")
        ET.SubElement(assets,"material",name="city_finish_"+kind,texture="city_floor_"+kind,
                      texuniform="true",texrepeat=".45 .45",emission=".12",specular=".4" if kind=="marble" else ".16")
    for key in ("street_neon","street_living","directory","leisure"):
        ET.SubElement(assets,"texture",name="city_"+key,type="2d",file=f"city_{key}.png")
        ET.SubElement(assets,"material",name="city_"+key,texture="city_"+key,emission=".45",specular=".12")
    wood=Mesh();frame=Mesh()
    for y in np.linspace(-.18,.18,5):wood.box((0,y,.445),(.73,.038,.025))
    for z in (.65,.78,.91):wood.box((0,.22,z),(.73,.028,.048))
    frame.box((0,0,.405),(.71,.215,.015))
    for x in (-.56,.56):
        for y in (-.16,.16):frame.cylinder((x,y,.025),(x,y,.43),.027)
        frame.cylinder((x,.20,.4),(x,.22,.97),.025)
        frame.cylinder((x,-.16,.64),(x,.23,.64),.025)
        frame.cylinder((x,-.16,.44),(x,-.16,.64),.021)
    wood.asset(assets,"city_bench_wood");frame.asset(assets,"city_bench_frame")
    grate=Mesh()
    for side in (-1,1):
        grate.box((side*.49,0,.315),(.035,.26,.02))
        grate.box((0,side*.24,.315),(.49,.025,.02))
    for x in np.linspace(-.42,.42,9):
        if abs(x)>.15:grate.box((x,0,.315),(.012,.22,.012))
    grate.asset(assets,"city_tree_grate")


def bench(parent,name,x,y,yaw=0):
    body=ET.SubElement(parent,"body",name=name,pos=numbers((x,y,0)),quat=numbers((np.cos(yaw/2),0,0,np.sin(yaw/2))))
    geom(body,name+"_seat",pos=(0,0,.43),size=(.73,.24,.04),material="walnut",rgba="0 0 0 0")
    for i,z in enumerate((.65,.78,.91)):
        geom(body,name+f"_back_collision_{i}",pos=(0,.22,z),size=(.73,.028,.048),material="walnut",rgba="0 0 0 0")
    for side in (-1,1):
        geom(body,name+f"_feet_{side}",pos=(side*.56,0,.215),size=(.035,.19,.215),material="steel",rgba="0 0 0 0")
    geom(body,name+"_slats","mesh",material="oak",solid=False,mesh="city_bench_wood")
    geom(body,name+"_frame","mesh",material="steel",solid=False,mesh="city_bench_frame")
    beam(body,name+"_underlight",(-.61,-.225,.36),(.61,-.225,.36),.012,"warm")
    return body


def street_sign(world,name,x,y,road,side):
    body=ET.SubElement(world,"body",name=name,pos=numbers((x,y,0)),quat="1 0 0 0" if side>0 else "0 0 0 1")
    geom(body,name+"_post","cylinder",(0,0,1.35),(.035,1.35),"steel")
    panel(body,name+"_nameplate","street_"+road,(.40,0,2.37),1.5,.47)
    geom(body,name+"_finial","sphere",(0,0,2.73),(.055,),"brass",False)


def add_street_furniture(assets,world,builder):
    shared_finishes(assets)
    # Replace the isolated old flower boxes with continuous, consistently spaced rows.
    for g in list(world.findall("geom")):
        if g.get("name","").startswith("street_planter_"):world.remove(g)
    for road in STREETS:
        cy=road["y"];offset=road["half_width"]+1.4
        for side in (-1,1):
            for i,x in enumerate(ROW_TREE_X):
                if road["id"]=="neon" and side<0:
                    x={2:-12.2,4:3.0,5:8.1}.get(i,x)
                y=cy+side*offset;name=f"avenue_tree_{road['id']}_{side}_{i}"
                geom(world,name+"_planter",pos=(x,y,.15),size=(.52,.24,.15),material="walnut")
                geom(world,name+"_soil",pos=(x,y,.307),size=(.46,.20,.012),material="soil",solid=False)
                geom(world,name+"_grate","mesh",pos=(x,y,0),material="steel",solid=False,mesh="city_tree_grate")
                tree(world,name,x,y,base=.33,height=3.15 if i%2 else 3.35,seed=820+i+side*16+(100 if cy else 0),autumn=(i+(side>0))%3==0)
            for i,x in enumerate(ROW_BENCH_X):
                bench(world,f"street_bench_{road['id']}_{side}_{i}",x,cy+side*(offset+.05),0 if side>0 else np.pi)
            for i,x in enumerate((-21.8,12.0)):
                street_sign(world,f"wayfinding_{road['id']}_{side}_{i}",x,cy+side*(offset+.04),road["id"],side)
            for i,x in enumerate((-24.8,23.8)):
                y=cy+side*(offset+.02);name=f"litter_{road['id']}_{side}_{i}"
                geom(world,name+"_body",pos=(x,y,.31),size=(.23,.17,.31),material="steel")
                for j,mat in enumerate(("teal_paint","brass")):
                    geom(world,name+f"_lid_{j}",pos=(x-.115+j*.23,y,.64),size=(.112,.175,.025),material=mat,solid=False)
                    geom(world,name+f"_opening_{j}",pos=(x-.115+j*.23,y,.669),size=(.055,.08,.004),material="ink",solid=False)
    for i,(x,y) in enumerate(((-19.0,15.5),(13.1,13.5))):
        name=f"district_directory_{i}"
        body=ET.SubElement(world,"body",name=name,pos=numbers((x,y,0)))
        geom(body,name+"_base",pos=(0,0,.14),size=(.58,.24,.14),material="walnut")
        for side in (-1,1):geom(body,name+f"_support_{side}","cylinder",(side*.53,0,1.14),(.035,1.14),"steel")
        geom(body,name+"_board",pos=(0,0,1.55),size=(.66,.06,.66),material="steel")
        panel(body,name+"_map","directory",(0,-.065,1.55),1.26,1.26)
    for i,x in enumerate((-25.,-11.,11.,25.)):
        old=world.find(f"geom[@name='living_park_bench_{i}']")
        if old is not None:world.remove(old)
        bench(world,f"plaza_bench_{i}",x+1.7,46.)
    park=world.find("body[@name='venue_park']")
    for side in (-1,1):bench(park,f"garden_bench_{side}",side*1.6,1.0,np.pi)
    geom(world,"leisure_table_top",pos=(1.5,45.5,.76),size=(.5,.5,.045),material="walnut")
    geom(world,"leisure_table_cards",pos=(1.5,45.5,.81),size=(.46,.46,.004),material="leisure",solid=False)
    geom(world,"leisure_table_pedestal","cylinder",(1.5,45.5,.36),(.12,.36),"steel")
    for side in (-1,1):bench(world,f"leisure_bench_{side}",1.5,45.5+side*1.2,0 if side>0 else np.pi)
