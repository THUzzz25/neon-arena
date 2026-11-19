"""Architectural detailing, prop bevels and readable street-level material layers."""
import itertools
import xml.etree.ElementTree as ET
import numpy as np

from .city import geom,beam,ring,numbers,VENUES,hiding_spots,world_position


class Details:
    def __init__(self,assets):
        self.assets=assets
        self.meshes={}

    def bevel(self,parent,name,pos,size,material="metal",bevel=.035,group=3):
        key=tuple(round(float(v),5) for v in (*size,bevel))
        if key not in self.meshes:
            mesh_name="city_bevel_"+str(len(self.meshes))
            # Truncated cube, with a small outward allowance over the collision proxy.
            half=np.asarray(size)+.016
            cut=min(bevel,float(min(half))*.55)
            vertices=[]
            for signs in itertools.product((-1,1),repeat=3):
                for axis in range(3):
                    vertex=half.copy();vertex[axis]-=cut
                    vertices.extend(vertex*np.array(signs))
            ET.SubElement(self.assets,"mesh",name=mesh_name,vertex=numbers(vertices),smoothnormal="false")
            self.meshes[key]=mesh_name
        return geom(parent,name,"mesh",pos=pos,size=(1,1,1),material=material,solid=False,group=group,mesh=self.meshes[key])


def refine_city(assets,world,builder):
    materials={
        "ceramic":((.43,.47,.59,1),.05,.45),
        "titanium":((.29,.33,.44,1),.05,.7),
        "warm":((1,.77,.47,1),.55,.3),
        "white":((.58,.82,1,1),.65,.25),
        "copper":((.53,.30,.24,1),.05,.6),
        "ink":((.015,.025,.05,1),.05,.2),
    }
    for key,(rgba,emission,specular) in materials.items():
        ET.SubElement(assets,"material",name="city_"+key,rgba=numbers(rgba),emission=str(emission),specular=str(specular),shininess=".65")
    paving=assets.find("material[@name='city_paving']")
    paving.set("rgba",".68 .74 .87 1")
    wall=assets.find("material[@name='city_wall']")
    wall.set("rgba",".23 .26 .37 1")
    wall.set("specular",".15");wall.set("shininess",".2")
    details=Details(assets)
    for venue in VENUES:
        key=venue["id"];w=venue["width"];d=venue["depth"]
        parent=world.find(f"body[@name='venue_{key}']")
        if key!="park":
            # Broad awning over open frontage, with a white underside light.
            details.bevel(parent,key+"_awning",(0,-.53,2.60),(w/2+.15,.6,.075),"titanium",.03,5)
            beam(parent,key+"_awning_light",(-w/2+.15,-1.1,2.59),(w/2-.15,-1.1,2.59),.024,"purple" if key in ("cafe","chapel") else "cyan")
            for x in np.linspace(-w/2+.3,w/2-.3,7):
                beam(parent,f"{key}_awning_rib_{x}",(x,-.98,2.52),(x,-.05,2.52),.018,"metal")
            for side in (-1,1):
                center=side*(w/4+.7);width=w/4-.8
                for offset in (-.50,0,.50):
                    u=center+offset*width
                    beam(parent,f"{key}_mullion_{side}_{offset}",(u,-.19,.79),(u,-.19,2.22),.017,"titanium")
                beam(parent,f"{key}_sill_{side}",(center-width,-.20,.78),(center+width,-.20,.78),.035,"metal")
                details.bevel(parent,f"{key}_corner_panel_{side}",(side*(w/2-.13),-.17,1.45),(.12,.05,1.38),"titanium",.03)
                for z in (.25,1.05,2.15):
                    geom(parent,f"{key}_bolt_{side}_{z}","sphere",(side*1.4,-.23,z),(.028,),"ceramic",False)
            # Wall seams and interior luminaires provide depth without narrowing aisles.
            for z in (.32,1.35,2.45):
                beam(parent,f"{key}_back_seam_{z}",(-w/2+.18,d-.15,z),(w/2-.18,d-.15,z),.008,"titanium")
            for u in (-w/2+.45,w/2-.45):
                beam(parent,f"{key}_interior_light_{u}",(u,.35,2.92),(u,d-.35,2.92),.026,"white")
            for u in np.arange(-w/2+.8,w/2,.95):
                beam(parent,f"{key}_back_joint_{u}",(u,d-.151,.2),(u,d-.151,2.65),.006,"panel")
            if key!="chapel":
                # Recessed upper floor: a second facade layer behind the ground-floor roof.
                geom(parent,key+"_upper_floor",pos=(0,d*.66,4.23),size=(w/2-.28,d*.33, .65),material="panel",solid=True,group=5)
                for side in (-1,1):
                    details.bevel(parent,f"{key}_upper_edge_{side}",(side*(w/2-.28),d*.66,4.22),(.09,d*.33,.69),"titanium",.035,5)
                for u in np.linspace(-w/2+.7,w/2-.7,5):
                    geom(parent,f"{key}_upper_window_{u}",pos=(u,d*.33-.02,4.25),size=(.34,.015,.34),material="glass",solid=False,group=5)
                    beam(parent,f"{key}_upper_window_neon_{u}",(u-.32,d*.33-.04,3.94),(u+.32,d*.33-.04,3.94),.018,"cyan").set("group","5")
                details.bevel(parent,key+"_upper_roof",(0,d*.66,4.95),(w/2-.1,d*.33+.15,.08),"titanium",.03,5)
                for u in (-1.4,1.4):
                    details.bevel(parent,f"{key}_vent_{u}",(u,d*.7,5.16),(.4,.3,.16),"metal",.045,5)
                    for j in range(5):
                        beam(parent,f"{key}_vent_louver_{u}_{j}",(u-.3,d*.7-.22+j*.1,5.33),(u+.3,d*.7-.22+j*.1,5.33),.012,"panel")
            # Entry sill is a texture-like thin plate; no raised doorstep.
            geom(parent,key+"_threshold",pos=(0,-.15,.006),size=(1.3,.23,.004),material="titanium",solid=False)
            for u in (-1.,-.5,0,.5,1.):
                beam(parent,f"{key}_threshold_tick_{u}",(u,-.3,.012),(u,-.02,.012),.013,"cyan")
        # Give storage modules bevelled skins, inset panels and packages.
        for index,spot in enumerate(s for s in hiding_spots() if s["zone"]==key):
            p=np.array(spot["position"]);origin=world_position(venue,0,0)
            u=(p[0]-origin[0])*venue["side"];v=(p[1]-origin[1])*venue["side"]
            z=p[2]-.18
            details.bevel(parent,f"{key}_storage_skin_{index}",(u,v,z/2),(.38,.28,z/2),"titanium",.035)
            geom(parent,f"{key}_storage_inset_{index}",pos=(u,v-.296,z*.52),size=(.27,.01,z*.27),material="panel",solid=False)
            beam(parent,f"{key}_storage_handle_{index}",(u-.12,v-.315,z*.8),(u+.12,v-.315,z*.8),.013,"ceramic")
            search_u=(spot["search_position"][0]-origin[0])*venue["side"]
            outward=-1 if search_u>u else 1
            for package in range(2):
                details.bevel(parent,f"{key}_package_{index}_{package}",(u+outward*.25,v+.08,z+.13+package*.14),(.075,.12,.065),"copper" if package else "panel",.018)
                beam(parent,f"{key}_package_band_{index}_{package}",(u+outward*.25-.06,v-.045,z+.15+package*.14),(u+outward*.25+.06,v-.045,z+.15+package*.14),.012,"amber")
        if key=="chapel":
            # Actual pitched roof, vaulted ribs and rose-window detail.
            old=parent.find("geom[@name='chapel_roof']")
            if old is not None:parent.remove(old)
            angle=np.arctan2(2.2,w/2);length=np.hypot(w/2,2.2)
            for side in (-1,1):
                geom(parent,f"chapel_sloped_roof_{side}",pos=(side*w/4,d/2,4.7),size=(length/2,d/2+.2,.10),material="panel",solid=True,group=5,euler=(0,side*angle,0))
            vertices=[]
            for y in (d-.10,d+.10):
                vertices.extend((-w/2,y,3.28));vertices.extend((w/2,y,3.28));vertices.extend((0,y,5.8))
            ET.SubElement(assets,"mesh",name="city_chapel_gable",vertex=numbers(vertices),smoothnormal="false")
            geom(parent,"chapel_back_gable","mesh",pos=(0,0,0),size=(1,1,1),material="wall",solid=True,group=5,mesh="city_chapel_gable")
            for v in (.25,2.2,4.4,d-.12):
                beam(parent,f"chapel_vault_left_{v}",(-w/2+.25,v,3.25),(0,v,5.55),.07,"titanium")
                beam(parent,f"chapel_vault_right_{v}",(w/2-.25,v,3.25),(0,v,5.55),.07,"titanium")
            ring(parent,"chapel_rose",(0,d-.17,3.75),.78,"purple",40,.035,tilt=np.pi/2)
            for angle in np.arange(0,2*np.pi,np.pi/4):
                beam(parent,f"chapel_rose_spoke_{angle}",(0,d-.17,3.75),(.74*np.cos(angle),d-.17,3.75+.74*np.sin(angle)),.025,"cyan")
            for v in (1.4,3.7):
                for side in (-1,1):
                    details.bevel(parent,f"chapel_seat_cushion_{v}_{side}",(side*.9,v,.65),(.34,.21,.055),"seat",.025)
                    for offset in (-.27,.27):
                        beam(parent,f"chapel_seat_arm_{v}_{side}_{offset}",(side*.9+offset,v-.12,.85),(side*.9+offset,v+.18,.85),.023,"titanium")
        elif key=="cafe":
            for x in (-1.7,1.65):
                ring(parent,f"cafe_table_rim_{x}",(x,3.5,.77),.42,"pink",24,.013)
                details.bevel(parent,f"cafe_chair_back_{x}",(x,2.68,.67),(.23,.04,.32),"seat",.04)
                geom(parent,f"cafe_coffee_plate_{x}","cylinder",(x,3.5,.79),(.12,.015),"ceramic",False)
                geom(parent,f"cafe_coffee_cup_{x}","cylinder",(x,3.5,.86),(.065,.06),"copper",False)
        elif key=="charge":
            for x in (-2.,0.,2.):
                details.bevel(parent,f"charge_shell_{x}",(x,3.2,.7),(.30,.23,.7),"titanium",.04)
                for row in range(4):
                    geom(parent,f"charge_status_{x}_{row}",pos=(x-.12+row*.08,2.935,1.22),size=(.023,.018,.04+row*.02),material="mint",solid=False)
                for segment in range(10):
                    a,b=np.pi*segment/10,np.pi*(segment+1)/10
                    beam(parent,f"charge_cable_{x}_{segment}",(x+.35+.23*np.sin(a),3.2,.85-.65*np.cos(a/2)),(x+.35+.23*np.sin(b),3.2,.85-.65*np.cos(b/2)),.023,"ink")
        elif key=="park":
            for side in (-1,1):
                for v in (1.4,4.6):
                    u=side*3.7
                    for j in range(5):
                        a=j*2*np.pi/5
                        geom(parent,f"park_leaf_{side}_{v}_{j}","ellipsoid",(u+.36*np.cos(a),v+.36*np.sin(a),2.25+.16*np.cos(a*2)),(.39,.24,.32),"leaf",False,euler=(0,.22,a))
                    beam(parent,f"park_tree_circuit_{side}_{v}",(u-.08,v,.65),(u-.08,v,1.65),.018,"mint")
        elif key in ("market","lab"):
            for shelf in range(3):
                z=1.2+shelf*.38
                geom(parent,f"{key}_rear_book_shelf_{shelf}",pos=(-1.6,d-.23,z),size=(.65,.12,.025),material="titanium",solid=False)
                for book in range(6):
                    details.bevel(parent,f"{key}_book_{shelf}_{book}",(-2.15+book*.19,d-.24,z+.13),(.055,.08,.105),"copper" if book%3==0 else "panel",.012)
                    beam(parent,f"{key}_book_led_{shelf}_{book}",(-2.15+book*.19,d-.33,z+.04),(-2.15+book*.19,d-.33,z+.21),.008,"cyan")
    # Drainage covers and lane-direction arrows give the pavement a human scale.
    for x in np.arange(-18,19,3.8):
        for side in (-1,1):
            for j in range(6):
                geom(world,f"drain_{x}_{side}_{j}",pos=(x-.25+j*.1,side*2.54,.007),size=(.018,.12,.005),material="ink",solid=False)
    for x in (-13,-5,3,11):
        for lane,direction in ((-1.25,1),(1.25,-1)):
            beam(world,f"road_arrow_stem_{x}_{lane}",(x-direction*.55,lane,.012),(x+direction*.25,lane,.012),.025,"white")
            for side in (-1,1):
                beam(world,f"road_arrow_head_{x}_{lane}_{side}",(x+direction*.25,lane,.012),(x-direction*.04,lane+side*.22,.012),.025,"white")
    for x in (-17.5,-5.7,5.4,17):
        for side in (-1,1):
            for j in range(8):
                a,b=j/8,(j+1)/8
                p=(x-1.4+2.8*a,side*5.02,4.12-.30*np.sin(a*np.pi))
                q=(x-1.4+2.8*b,side*5.02,4.12-.30*np.sin(b*np.pi))
                beam(world,f"lamp_cable_{x}_{side}_{j}",p,q,.012,"ink")
    # Hovercraft: bevelled body shell, turbine rings, radiator slats and antenna.
    for body in world.findall("body"):
        name=body.get("name","")
        if name.startswith("traffic/"):
            details.bevel(body,name+"/bodywork",(0,0,0),(1.25,.58,.18),"titanium",.07)
            for side in (-1,1):
                for j in range(6):
                    beam(body,name+f"/radiator_{side}_{j}",(-.9+j*.14,side*.605,-.06),(-.9+j*.14,side*.605,.09),.018,"ink")
                for x in (-.85,.85):
                    ring(body,name+f"/turbine_ring_{side}_{x}",(x,side*.42,-.34),.16,"cyan",20,.014)
            beam(body,name+"/antenna",(-.65,0,.55),(-.8,0,.86),.013,"metal")
            geom(body,name+"/antenna_tip","sphere",(-.8,0,.86),(.026,),"pink",False)
        elif name.startswith("citizen/") and name.count("/")==1:
            details.bevel(body,name+"/chest_shell",(0,0,.25),(.2,.145,.24),"titanium",.055)
            for side in (-1,1):
                geom(body,name+f"/shoulder_led_{side}","sphere",(0,side*.17,.42),(.055,),"cyan",False)
