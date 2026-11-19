"""Room finishes, joinery, lighting and store-specific small furnishings."""
import numpy as np

from .city import VENUES,geom,beam,numbers,hiding_spots,world_position
from .city_meshes import Mesh
from .city_shops import panel

THEMES={
    "mixue":("terrazzo","ivory","shop_red"),"luckin":("timber","ivory","shop_blue"),
    "shaxian":("terrazzo","linen","shop_red"),"skewers":("concrete","linen","bronze"),
    "hardware":("concrete","ivory","olive"),"hotel":("marble","ivory","brass"),
    "netcafe":("carpet","upholstery","shop_blue"),"tanghulu":("terrazzo","ivory","shop_red"),
    "convenience":("terrazzo","ivory","olive"),"laundry":("ceramic","ivory","teal_paint"),
}


def prepare_details(assets):
    plant=Mesh();rng=np.random.default_rng(941)
    for i in range(48):
        angle=i*2.4;center=np.array([.11*np.cos(angle),.11*np.sin(angle),.13+.006*i])
        plant.leaf(center,angle,.23+rng.uniform(0,.13),.034,rng.uniform(.2,1.5))
    plant.asset(assets,"city_indoor_fern")
    for radius in (.08,.12,.25,.30):
        m=Mesh();n=32;k=6;points=[];faces=[]
        for i in range(n):
            a=i*2*np.pi/n
            for j in range(k):
                b=j*2*np.pi/k
                points.append(((radius+.012*np.cos(b))*np.cos(a),(radius+.012*np.cos(b))*np.sin(a),.012*np.sin(b)))
        for i in range(n):
            for j in range(k):
                a=i*k+j;b=((i+1)%n)*k+j;c=((i+1)%n)*k+(j+1)%k;d=i*k+(j+1)%k
                faces.extend(((a,b,c),(a,c,d)))
        m.add(points,faces);m.asset(assets,f"city_detail_ring_{round(radius*100)}")


def ring_detail(parent,name,pos,radius,mat="brass",euler=(0,0,0)):
    geom(parent,name,"mesh",pos=pos,material=mat,solid=False,mesh=f"city_detail_ring_{round(radius*100)}",euler=euler)


def plant(parent,name,u,v):
    geom(parent,name+"_pot","cylinder",(u,v,.16),(.17,.16),"ivory")
    geom(parent,name+"_soil","cylinder",(u,v,.325),(.145,.012),"soil",False)
    geom(parent,name+"_fern","mesh",(u,v,.33),(1,1,1),"leaf_light",False,mesh="city_indoor_fern")


def pendant(parent,name,u,v):
    beam(parent,name+"_wire",(u,v,2.64),(u,v,3.28),.009,"steel").set("group","5")
    geom(parent,name+"_shade","cylinder",(u,v,2.58),(.18,.07),"walnut",False,5)
    geom(parent,name+"_diffuser","cylinder",(u,v,2.502),(.15,.012),"warm",False,5)


def common_room(assets,world,venue,kind):
    key=venue["id"];w,d=venue["width"],venue["depth"];parent=world.find(f"body[@name='venue_{key}']")
    floor,wall,accent=THEMES[kind]
    parent.find(f"geom[@name='{key}_floor']").set("material","city_finish_"+floor)
    for g in parent.findall("geom"):
        name=g.get("name","")
        if name in (key+"_back",key+"_side_-1",key+"_side_1"):g.set("material","city_"+wall)
        elif "_back_joint_" in name or "_back_seam_" in name:g.set("material","city_steel")
        elif "_storage_skin_" in name:g.set("material","city_"+("walnut" if kind in ("hotel","luckin","skewers") else accent))
        elif "_storage_handle_" in name:g.set("material","city_brass")
        elif name.startswith(key+"_screen_"):g.set("material","city_"+("upholstery" if kind=="netcafe" else "walnut" if kind in ("hotel","luckin") else "linen"))
    # Wall joinery uses the existing shell, leaving the floor plan intact.
    geom(parent,key+"_back_wainscot",pos=(0,d-.156,.53),size=(w/2-.19,.012,.48),material="walnut",solid=False)
    for side in (-1,1):
        geom(parent,key+f"_side_wainscot_{side}",pos=(side*(w/2-.156),d/2,.53),size=(.012,d/2-.18,.48),material="walnut",solid=False)
        beam(parent,key+f"_dado_rail_{side}",(side*(w/2-.18),.2,1.04),(side*(w/2-.18),d-.18,1.04),.014,"brass")
    beam(parent,key+"_back_dado",(-w/2+.2,d-.18,1.04),(w/2-.2,d-.18,1.04),.014,"brass")
    panel(parent,key+"_feature_wall","sign_"+venue.get("sign",key),(1.0 if key=="hotel" else 0,d-.18,2.27),2.5,.65)
    geom(parent,key+"_ceiling_light",pos=(0,d*.46,3.29),size=(.55,.19,.018),material="warm" if kind not in ("netcafe","laundry") else "white",solid=False,group=5)
    # Trim makes the hiding screens read as shop dividers instead of bare slabs.
    frame=Mesh()
    for i,spot in enumerate(s for s in hiding_spots() if s["zone"]==key):
        origin=world_position(venue,0,0);p=np.asarray(spot["position"])
        u=(p[0]-origin[0])*venue["side"];v=(p[1]-origin[1])*venue["side"]-.6
        for side in (-1,1):frame.box((u+side*.405,v-.09,.78),(.018,.018,.78))
        for z in (.08,.77,1.49):frame.box((u,v-.092,z),(.40,.018,.014))
    frame.asset(assets,"city_divider_trim_"+key)
    geom(parent,key+"_divider_trim","mesh",material="brass" if kind in ("hotel","hardware") else "oak",solid=False,mesh="city_divider_trim_"+key)
    plant(parent,key+"_indoor_plant",w/2-.65,2.8)
    return parent


def refine_seating(parent,key):
    for g in list(parent.findall("geom")):
        name=g.get("name","")
        if "_stool_" not in name or g.get("type")!="cylinder":continue
        x,y,_=map(float,g.get("pos").split())
        g.set("pos",numbers((x,y,.435)));g.set("size",".19 .035")
        for sx in (-1,1):
            for sy in (-1,1):
                beam(parent,name+f"_leg_{sx}_{sy}",(x+sx*.12,y+sy*.12,.02),(x+sx*.12,y+sy*.12,.40),.017,"steel",True)
    if key=="cafe":
        for x in (-1.7,1.65):
            g=parent.find(f"geom[@name='cafe_chair_{x}']")
            g.set("pos",numbers((x,2.9,.44)));g.set("size",".23 .23 .04");g.set("material","city_upholstery")
            for sx in (-1,1):
                for sy in (-1,1):beam(parent,f"cafe_chair_legs_{x}_{sx}_{sy}",(x+sx*.16,2.9+sy*.16,.02),(x+sx*.16,2.9+sy*.16,.40),.018,"steel",True)


def counter_detail(parent,key,u,v,accent):
    geom(parent,key+"_counter_cap",pos=(u,v,1.005),size=(.41,.36,.025),material="walnut",solid=False)
    for i,z in enumerate((.27,.57,.84)):
        geom(parent,key+f"_drawer_{i}",pos=(u,v-.347,z),size=(.34,.008,.11),material=accent,solid=False)
        beam(parent,key+f"_pull_{i}",(u-.11,v-.365,z+.035),(u+.11,v-.365,z+.035),.012,"brass")


def dining_details(parent,key,u,v,z=.81):
    ring_detail(parent,key+"_plate_rim",(u-.12,v,z),.12,"ivory")
    geom(parent,key+"_plate","ellipsoid",(u-.12,v,z-.008),(.12,.12,.015),"ivory",False)
    geom(parent,key+"_napkin_box",pos=(u+.13,v+.12,z+.08),size=(.085,.045,.075),material="oak",solid=False)
    geom(parent,key+"_napkin",pos=(u+.13,v+.12,z+.165),size=(.065,.005,.035),material="shop_white",solid=False)
    for i in range(2):
        geom(parent,key+f"_condiment_{i}","cylinder",(u+.13+i*.11,v-.13,z+.055),(.028,.055),"shop_red" if i else "steel",False)


def refine_store(assets,world,venue):
    key=venue["id"];kind=venue.get("shop",{"noodle":"skewers","arcade":"netcafe"}.get(key))
    if kind is None:return
    p=common_room(assets,world,venue,kind);w,d=venue["width"],venue["depth"];left=-w/2+.75
    refine_seating(p,key)
    if p.find(f"geom[@name='{key}_sales_counter']") is not None:
        counter_detail(p,key,left,2.8,THEMES[kind][2])
        pendant(p,key+"_counter_pendant",left,2.8)
    if kind in ("mixue","luckin"):
        if key=="cafe":
            left=-2.65
            geom(p,"cafe_barista_counter",pos=(left,.95,.5),size=(.42,.34,.5),material="walnut")
            geom(p,"cafe_barista_machine",pos=(left,1.04,1.30),size=(.29,.18,.28),material="steel",solid=False)
            cy=.95
        elif key=="tea":
            left=-3.45;cy=3.4
            geom(p,"tea_tea_dispenser",pos=(left,cy+.1,1.3),size=(.3,.19,.28),material="shop_white",solid=False)
        else:cy=2.8
        for j in range(3):
            x=left-.18+j*.18
            beam(p,key+f"_tap_{j}",(x,cy-.1,1.36),(x,cy-.29,1.36),.017,"steel")
            geom(p,key+f"_tap_lever_{j}",pos=(x,cy-.16,1.47),size=(.019,.025,.065),material="shop_blue" if kind=="luckin" else "shop_red",solid=False)
        geom(p,key+"_machine_display",pos=(left,cy-.097,1.55),size=(.115,.012,.06),material="mint",solid=False)
        if kind=="luckin":
            geom(p,key+"_bean_hopper","ellipsoid",(left,cy+.1,1.72),(.15,.13,.16),"glass",False,rgba=".32 .24 .16 .6")
        if key not in ("cafe","tea"):dining_details(p,key+"_cafe_place",.4,2.9)
        if key=="cafe":dining_details(p,"cafe_table_place",-1.7,3.5,.78)
    elif kind=="shaxian":
        if key=="shaxian":
            for j,z in enumerate((1.01,1.11,1.21)):ring_detail(p,key+f"_bamboo_rim_{j}",(left,2.8,z),.25,"oak")
            dining_details(p,key+"_meal",.4,2.8)
        else:
            dining_details(p,key+"_meal",-.15,2.5,1.04)
        geom(p,key+"_service_shelf",pos=(-w/2+.55,d-.4,1.6),size=(.4,.17,.035),material="oak",solid=False)
        for j in range(3):
            geom(p,key+f"_bowl_stack_{j}","cylinder",(-w/2+.55,d-.4,1.67+j*.055),(.11,.022),"shop_white",False)
    elif kind=="skewers" and key=="skewers":
        geom(p,key+"_exhaust_duct",pos=(left,2.8,2.93),size=(.18,.20,.39),material="steel",solid=False,group=5)
        for i in range(5):beam(p,key+f"_hood_slit_{i}",(left-.35+i*.17,2.35,2.34),(left-.35+i*.17,2.35,2.43),.012,"ink")
        dining_details(p,key+"_night_place",.5,2.8)
    elif kind=="tanghulu":
        for side in (-1,1):
            for depth in (-1,1):beam(p,key+f"_stall_post_{side}_{depth}",(left+side*.46,2.8+depth*.40,1.),(left+side*.46,2.8+depth*.40,2.22),.018,"brass")
        panel(p,key+"_stall_sign","sign_tanghulu",(left,2.3,2.23),1.02,.23)
        for j in range(3):
            geom(p,key+f"_wrapping_{j}",pos=(left-.24+j*.22,3.05,1.06),size=(.08,.12,.025),material="ivory",solid=False)
    elif kind=="hardware":
        bins=Mesh()
        for j in range(5):
            u=left-.30+j*.15
            bins.box((u,2.61,1.04),(.068,.12,.025))
            for k in range(3):bins.cylinder((u-.04+k*.035,2.61,1.065),(u-.04+k*.035,2.61,1.11),.012,sides=6)
        bins.asset(assets,"city_hardware_small_parts")
        geom(p,"hardware_parts_bins","mesh",material="steel",solid=False,mesh="city_hardware_small_parts")
        for j in range(4):
            geom(p,key+f"_safety_band_{j}",pos=(left-.29+j*.19,2.438,.88),size=(.055,.005,.04),material="gold" if j%2 else "ink",solid=False)
    elif kind=="hotel":
        bell=p.find("geom[@name='hotel_reception_bell']")
        bell.set("type","ellipsoid");bell.set("size",".10 .10 .045");bell.set("material","city_brass")
        geom(p,key+"_bell_base","cylinder",(-.7,2.82,1.06),(.11,.012),"brass",False)
        geom(p,key+"_bell_button","cylinder",(-.7,2.82,1.157),(.018,.02),"steel",False)
        geom(p,key+"_desk_slab",pos=(-1.1,3.,1.065),size=(1.14,.39,.025),material="walnut",solid=False)
        for u in (-1.8,-1.1,-.4):geom(p,key+f"_desk_inlay_{u}",pos=(u,2.64,.57),size=(.006,.008,.36),material="brass",solid=False)
        panel(p,key+"_desk_terminal","desktop",(-1.65,3.1,1.34),.45,.27)
        for j in range(3):geom(p,key+f"_key_card_{j}",pos=(-.15+j*.1,2.89,1.10),size=(.035,.055,.005),material="ivory",solid=False)
    elif kind=="netcafe" and key=="netcafe":
        desk=p.find("geom[@name='netcafe_pc_desk']");desk.set("pos","0 2.9 .84");desk.set("size","1.78 .33 .045")
        for x in (-1.62,1.62):
            for y in (2.66,3.14):beam(p,key+f"_desk_leg_{x}_{y}",(x,y,.02),(x,y,.80),.032,"steel",True)
        for j,u in enumerate((-1.2,0,1.2)):
            geom(p,key+f"_monitor_base_{j}",pos=(u,2.95,.90),size=(.16,.10,.012),material="steel",solid=False)
            beam(p,key+f"_monitor_stand_{j}",(u,2.95,.9),(u,2.95,1.02),.018,"steel")
            geom(p,key+f"_mouse_{j}","ellipsoid",(u+.35,2.67,.90),(.045,.065,.025),"steel",False)
        geom(p,key+"_tower",pos=(-1.35,2.9,.35),size=(.20,.28,.35),material="ink")
        for z in (.24,.49):ring_detail(p,key+f"_tower_fan_{z}",(-1.35,2.607,z),.08,"mint",(np.pi/2,0,0))
        beam(p,key+"_desk_rgb",(-1.7,2.555,.80),(1.7,2.555,.80),.012,"purple")
        for j,u in enumerate((-.6,.6)):
            geom(p,key+f"_chair_headrest_{j}",pos=(u,1.78,1.02),size=(.13,.055,.075),material="upholstery",solid=False)
            for side in (-1,1):beam(p,key+f"_armrest_{j}_{side}",(u+side*.22,1.80,.69),(u+side*.22,2.10,.69),.022,"steel")
    elif kind=="convenience":
        geom(p,key+"_fridge_handle",pos=(2.86,2.335,1.03),size=(.012,.018,.32),material="steel",solid=False)
        for j in range(3):geom(p,key+f"_fridge_grille_{j}",pos=(2.6,2.332,.16+j*.045),size=(.26,.008,.008),material="steel",solid=False)
    elif kind=="laundry":
        for j,u in enumerate((-1.4,0,1.4)):
            ring_detail(p,key+f"_door_bezel_{j}",(u,2.805,.58),.30,"steel",(np.pi/2,0,0))
            geom(p,key+f"_dial_{j}","cylinder",(u-.25,2.87,1.0),(.065,.025),"steel",False,euler=(np.pi/2,0,0))
            for k in range(3):geom(p,key+f"_folded_towel_{j}_{k}",pos=(u,3.3,1.19+k*.045),size=(.25,.17,.02),material="ivory" if k%2 else "teal_paint",solid=False)


def refine_interiors(assets,world):
    prepare_details(assets)
    for venue in VENUES:refine_store(assets,world,venue)
