"""Everyday businesses and the second, warmer commercial street."""
import xml.etree.ElementTree as ET
import numpy as np

from .city import VENUES,CONNECTORS,geom,beam,numbers
from .city_art import SHOP_SIGNS
from .city_life import material,tree,lantern,tower,advert


def panel(parent,name,texture,pos,width,height,group=3):
    x,y,z=pos
    geom(parent,name+"_frame",pos=pos,size=(width/2+.04,.035,height/2+.04),material="ink",solid=False,group=group)
    geom(parent,name+"_image",pos=(x,y-.045,z),size=(width/2,height/2,.005),material=texture,solid=False,group=group,xyaxes=(1,0,0,0,0,1))


def cup(parent,name,u,v,z=1.07,iced=False):
    geom(parent,name,"cylinder",(u,v,z),(.065,.085),"shop_white",False)
    geom(parent,name+"_lid","cylinder",(u,v,z+.09),(.071,.012),"shop_blue" if iced else "bronze",False)
    if iced:beam(parent,name+"_straw",(u,v,z+.1),(u+.02,v,z+.26),.009,"red")


def table(parent,name,u,v):
    geom(parent,name+"_top","cylinder",(u,v,.76),(.35,.035),"shop_wood")
    geom(parent,name+"_leg","cylinder",(u,v,.37),(.06,.37),"bronze")
    for side in (-1,1):
        geom(parent,name+f"_stool_{side}","cylinder",(u+side*.63,v,.22),(.18,.22),"shop_red")


def sweets(parent,name,u,v,z=1.05):
    for j in range(3):
        x=u+(j-1)*.16
        beam(parent,name+f"_stick_{j}",(x,v,z-.08),(x,v,z+.45),.01,"shop_wood")
        for k in range(3):
            geom(parent,name+f"_fruit_{j}_{k}","sphere",(x,v,z+.10+k*.11),(.065,),"shop_red",False)


def dress_shop(world,venue):
    key=venue["id"];kind=venue.get("shop")
    if kind is None:return
    parent=world.find(f"body[@name='venue_{key}']");w,d=venue["width"],venue["depth"]
    accent={"luckin":"shop_blue","mixue":"shop_red","shaxian":"shop_red","skewers":"bronze",
        "hardware":"olive","hotel":"bronze","netcafe":"shop_blue","tanghulu":"shop_red",
        "convenience":"olive","laundry":"shop_teal"}[kind]
    for g in parent.findall("geom"):
        n=g.get("name","")
        if n in (key+"_back",key+"_lintel",key+"_side_-1",key+"_side_1"):
            g.set("material","city_shop_cream")
        elif n.startswith(key+"_front_") and "glass" not in n:
            g.set("material","city_"+accent)
        elif "portal" in n or n==key+"_top_neon":
            g.set("material","city_"+("gold" if kind in ("hotel","shaxian","skewers","hardware") else "white"))
        elif n==key+"_awning":g.set("material","city_shop_cream")
    panel(parent,key+"_daily_menu","menu_"+kind,(w/2-.6,-.23,1.55),.85,1.32)
    badge=parent.find(f"body[@name='tower_ad_{key}']")
    if badge is not None:
        badge.find(f"geom[@name='tower_ad_{key}_screen']").set("material","city_menu_"+kind)
    # Existing south-street stores retain their established aisles and furniture.
    if key in ("market","cafe","tea"):
        if key=="market":
            for g in list(parent.findall("geom")):
                if g.get("name","").startswith(("market_book_","market_rear_book_shelf_")):
                    parent.remove(g)
            for j in range(3):geom(parent,f"market_dumpling_tray_{j}","ellipsoid",(-.35+j*.25,2.5,1.055),(.11,.09,.03),"shop_white",False)
        elif key=="cafe":
            cup(parent,"cafe_pickup_cup",-1.7,3.5,.9,True)
        else:
            cup(parent,"tea_pickup_cup",3.45,3.4,1.08,True)
        return
    # All larger furniture is physical; the small products are visual detail.
    left=-w/2+.75
    if kind in ("mixue","luckin","shaxian","skewers","tanghulu","hardware","convenience"):
        geom(parent,key+"_sales_counter",pos=(left,2.8,.49),size=(.39,.34,.49),material=accent)
    if kind in ("mixue","luckin"):
        geom(parent,key+"_dispenser",pos=(left,2.95,1.32),size=(.28,.19,.28),material="shop_white",solid=False)
        for j in range(3):
            cup(parent,key+f"_pickup_{j}",left-.2+j*.2,2.6,1.08,True)
        table(parent,key+"_cafe_table",.4,2.9)
        cup(parent,key+"_table_cup",.4,2.9,.88,True)
        if kind=="mixue":
            geom(parent,key+"_cone","cylinder",(left,2.94,1.82),(.12,.14),"shop_wood",False)
            geom(parent,key+"_icecream","ellipsoid",(left,2.94,2.02),(.17,.17,.19),"shop_white",False)
    elif kind=="shaxian":
        table(parent,key+"_dining_table",.4,2.8)
        for j in range(3):geom(parent,key+f"_dumplings_{j}","ellipsoid",(.2+j*.17,2.8,.83),(.065,.045,.035),"shop_white",False)
        for j in range(2):
            geom(parent,key+f"_steamer_{j}","cylinder",(left,2.8,1.05+j*.11),(.25,.05),"shop_wood",False)
    elif kind=="skewers":
        geom(parent,key+"_coals",pos=(left,2.8,1.005),size=(.3,.25,.025),material="red",solid=False)
        for j in range(4):
            x=left-.24+j*.16
            beam(parent,key+f"_grill_bar_{j}",(x,2.42,1.04),(x,3.16,1.04),.014,"metal")
            for k in range(2):geom(parent,key+f"_meat_{j}_{k}",pos=(x,2.65+k*.2,1.085),size=(.055,.07,.035),material="copper",solid=False)
        geom(parent,key+"_extractor",pos=(left,2.8,2.37),size=(.55,.5,.12),material="bronze",solid=False)
        table(parent,key+"_night_table",.5,2.8)
    elif kind=="tanghulu":
        sweets(parent,key+"_display",left,2.7)
        geom(parent,key+"_candy_canopy",pos=(left,2.8,2.22),size=(.52,.48,.06),material="shop_red",solid=False)
    elif kind=="hardware":
        geom(parent,key+"_pegboard",pos=(left,3.2,1.65),size=(.55,.04,.48),material="shop_wood",solid=False)
        for j in range(4):
            x=left-.36+j*.24
            beam(parent,key+f"_tool_handle_{j}",(x,3.14,1.4),(x,3.14,1.83),.025,"metal")
            beam(parent,key+f"_tool_head_{j}",(x-.08,3.14,1.83),(x+.08,3.14,1.83),.04,"metal")
        for j in range(4):
            geom(parent,key+f"_pipe_{j}","cylinder",(-3.5+j*.16,d-.65,.65),(.045,.6),"copper",False)
        geom(parent,key+"_parts_crate",pos=(-2.8,d-.65,.3),size=(.4,.35,.3),material="bronze")
    elif kind=="hotel":
        geom(parent,key+"_reception_desk",pos=(-1.1,3.,.52),size=(1.1,.35,.52),material="shop_wood")
        geom(parent,key+"_reception_bell","sphere",(-.7,2.82,1.1),(.09,),"gold",False)
        panel(parent,key+"_welcome","menu_hotel",(-1.4,d-.18,1.85),1.1,1.55)
        for j in range(2):
            geom(parent,key+f"_luggage_{j}",pos=(-3.0+j*.58,d-.7,.35),size=(.23,.23,.35),material="shop_blue")
        tower(parent,"near_tower_hotel",0,d*.65,5.04,14.,3.35,1.55,3,5)
        panel(parent,"hotel_vertical_sign","menu_hotel",(-3.8,-.22,5.7),1.1,3.8,5)
    elif kind=="netcafe":
        geom(parent,key+"_pc_desk",pos=(0,2.9,.42),size=(1.78,.33,.42),material="ink")
        for j,u in enumerate((-1.2,0,1.2)):
            panel(parent,key+f"_monitor_{j}","desktop",(u,2.95,1.25),.8,.5)
            geom(parent,key+f"_keyboard_{j}",pos=(u,2.66,.87),size=(.28,.09,.018),material="shop_blue",solid=False)
        for j,u in enumerate((-.6,.6)):
            geom(parent,key+f"_chair_{j}","cylinder",(u,1.95,.24),(.18,.24),"shop_blue")
            geom(parent,key+f"_chair_back_{j}",pos=(u,1.78,.65),size=(.19,.045,.3),material="shop_blue",solid=False)
    elif kind=="convenience":
        for shelf in range(3):
            z=1.05+shelf*.28
            geom(parent,key+f"_shelf_{shelf}",pos=(left,2.8,z),size=(.36,.3,.025),material="shop_wood",solid=False)
            for j in range(4):
                geom(parent,key+f"_bottle_{shelf}_{j}","cylinder",(left-.23+j*.15,2.78,z+.12),(.045,.1),"amber" if j%2 else "mint",False)
        geom(parent,key+"_fridge",pos=(2.6,2.8,.85),size=(.38,.4,.85),material="shop_white")
        panel(parent,key+"_fridge_glass","ad_energy",(2.6,2.36,1.02),.63,1.15)
    elif kind=="laundry":
        for j,u in enumerate((-1.4,0,1.4)):
            geom(parent,key+f"_washer_{j}",pos=(u,3.3,.58),size=(.5,.4,.58),material="shop_white")
            geom(parent,key+f"_washer_door_{j}","cylinder",(u,2.88,.58),(.29,.025),"shop_blue",False,euler=(np.pi/2,0,0))
            geom(parent,key+f"_washer_window_{j}","cylinder",(u,2.84,.58),(.22,.015),"glass",False,euler=(np.pi/2,0,0))
            geom(parent,key+f"_washer_display_{j}",pos=(u+.22,2.88,.99),size=(.13,.015,.055),material="mint",solid=False)
        geom(parent,key+"_laundry_basket",pos=(-3.4,d-.8,.3),size=(.4,.3,.3),material="shop_wood")


def add_living_district(assets,world,builder):
    colors={"shop_cream":((.67,.55,.43,1),.15),"shop_red":((.63,.06,.075,1),.22),
        "shop_blue":((.05,.16,.43,1),.12),"shop_wood":((.39,.23,.13,1),.08),
        "shop_white":((.85,.83,.73,1),.22),"shop_teal":((.08,.4,.43,1),.15),
        "road_warm":((.08,.065,.065,1),.05)}
    for key,(rgba,emission) in colors.items():material(assets,key,rgba,emission)
    for key in SHOP_SIGNS:
        ET.SubElement(assets,"texture",name="city_menu_"+key,type="2d",file=f"city_menu_{key}.png")
        ET.SubElement(assets,"material",name="city_menu_"+key,texture="city_menu_"+key,emission=".6",specular=".1")
    for key,file in (("warm_paving","city_living_paving.png"),("desktop","city_desktop.png")):
        ET.SubElement(assets,"texture",name="city_"+key,type="2d",file=file)
        ET.SubElement(assets,"material",name="city_"+key,texture="city_"+key,texuniform="true" if key=="warm_paving" else "false",
                      texrepeat=".7 .7" if key=="warm_paving" else "1 1",emission=".15" if key=="warm_paving" else ".65")
    builder.box(world,"living_road",(0,30,-.019),(30.5,3.1,.018),"road_warm")
    crossings=(-24,-16.7,-8,0,8.1,15.6,24)
    for side in (-1,1):
        builder.box(world,f"living_sidewalk_{side}",(0,30+side*4.65,-.012),(30,1.55,.01),"warm_paving")
        beam(world,f"living_curb_light_{side}",(-30,30+side*3.22,.025),(30,30+side*3.22,.025),.018,"gold")
        start=-30
        for crossing in (*crossings,31.1):
            end=min(30,crossing-1.1)
            if end>start:builder.box(world,f"living_curb_{side}_{start}",((start+end)/2,30+side*3.13,.055),((end-start)/2,.09,.055),"bronze")
            start=crossing+1.1
    for x in np.arange(-29,30,3.):
        beam(world,f"living_lane_dash_{x}",(x,30,.016),(x+1.25,30,.016),.025,"amber")
    for x in crossings:
        for dy in np.arange(-2.8,2.9,.6):
            geom(world,f"living_crosswalk_{x}_{dy}",pos=(x,30+dy,.006),size=(.85,.18,.004),material="shop_white",solid=False)
    for x in (-24,-8,8,24):
        for lane,direction in ((28.5,1),(31.5,-1)):
            beam(world,f"living_arrow_{x}_{lane}",(x-.5*direction,lane,.014),(x+.3*direction,lane,.014),.025,"shop_white")
            for side in (-1,1):beam(world,f"living_arrow_head_{x}_{lane}_{side}",(x+.3*direction,lane,.014),(x,lane+side*.23,.014),.025,"shop_white")
    # Rear promenade and the open court keep the two shop rows from becoming a wall.
    builder.box(world,"north_prom_playground",(0,45.5,-.014),(30,3.6,.012),"warm_paving")
    builder.box(world,"middle_service_walk",(0,15.5,-.014),(30,1.5,.012),"paving")
    builder.box(world,"middle_pedestrian_link",(2.65,15.,-.014),(.7,15.,.012),"paving")
    for j,x in enumerate((-25.,-11.,11.,25.)):
        builder.planter(world,f"living_park_planter_{j}",x,46.,.65,.38)
        leaf=world.find(f"geom[@name='living_park_planter_{j}_leaves']")
        if leaf is not None:world.remove(leaf)
        tree(world,f"living_park_tree_{j}",x,46.,base=.52,height=2.9,seed=600+j,autumn=j%2==0)
        geom(world,f"living_park_bench_{j}",pos=(x+1.7,46.,.24),size=(.65,.25,.24),material="shop_wood")
    # Lamp strings and a recognisable neighbourhood entrance contrast with neon gantries.
    for x in (-6.8,16.8):
        for side in (-1,1):
            geom(world,f"living_string_post_{x}_{side}","cylinder",(x,30+side*5.7,3.1),(.065,3.1),"bronze")
        for j in range(8):
            y=24.5+j*11/7;z=5.5-.65*np.sin(j*np.pi/7)
            lantern(world,f"living_lantern_{x}_{j}",(x,y,z),j%2==0)
    banner=ET.SubElement(world,"body",name="living_banner",pos="-7.1 30 6.2",euler="0 0 -1.5707963")
    builder.sign(banner,"living_district_sign","living",(0,0,0),7.6,.9)
    beam(world,"living_banner_support",(-6.8,24.3,6.15),(-6.8,35.7,6.15),.065,"bronze")
    for side in (-1,1):
        advert(world,f"living_end_ad_{side}","delivery" if side<0 else "tea",(side*30.4,30,2.3),4.8,2.3,-side*np.pi/2)
    for venue in VENUES:dress_shop(world,venue)
    # Extra sweets on the original night market make daily shops present on both roads.
    market=world.find("body[@name='venue_noodle']")
    sweets(market,"old_market_tanghulu",-3.95,3.3)
