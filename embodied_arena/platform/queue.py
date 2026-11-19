from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import secrets
import re
import threading
import time
import traceback
import uuid

from .isolation import IsolatedPool, docker, probe
from .robot_contract import validate_manifest


class CompetitionQueue:
    def __init__(self,store,settings):
        self.store,self.settings=store,settings
        self.stop=threading.Event()
        self.wake=threading.Event()
        self.deploying=threading.Event()
        self.thread=None
        self.node={"ready":False,"environment":settings.environment,"message":"正在检测执行环境…","gpu":False}
        self.node_checked=0.

    def refresh(self):
        self.node=probe(self.settings)
        if self.node.get("ready"):
            key="image:"+self.settings.environment
            with self.store.connect(write=True) as db:
                db.execute("INSERT OR IGNORE INTO settings VALUES (?,?)",(key,self.node["image_id"]))
                pinned=db.execute("SELECT value FROM settings WHERE key=?",(key,)).fetchone()[0]
            if self.node["image_id"]!=pinned:
                self.node={**self.node,"ready":False,"message":"环境版本已固定到另一镜像摘要。请恢复原镜像；依赖更新需要发布新的环境版本。"}
        self.node_checked=time.monotonic()
        return self.node

    def start(self):
        self.store.execute("UPDATE jobs SET state='failed',message='服务器重启，原运行已中断，请重新发起',finished=? WHERE state IN ('preparing','running')",(time.time(),))
        self.store.execute("UPDATE submissions SET state='waiting_check' WHERE state='checking'")
        self.thread=threading.Thread(target=self.loop,name="arena-queue",daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set();self.wake.set()
        if self.thread:
            self.thread.join(timeout=20)

    def loop(self):
        # Only reap containers labelled for this installation after acquiring
        # the server lock. Never affect unrelated containers on the machine.
        try:
            old=docker(["ps","-aq","--filter",f"label=org.neon-arena.installation={self.store.setting('installation')}"],timeout=10)
            for container in old.stdout.split():
                if re.fullmatch(r"[0-9a-f]{12,64}",container):
                    docker(["rm","-f",container],timeout=10)
        except Exception:
            pass
        while not self.stop.is_set():
            try:
                if time.monotonic()-self.node_checked>15:
                    self.refresh()
                if self.node.get("ready") and not self.deploying.is_set():
                    with self.store.connect(write=True) as db:
                        row=None if self.deploying.is_set() else db.execute("SELECT * FROM jobs WHERE state IN ('queued','waiting_node') ORDER BY CASE kind WHEN 'official' THEN 0 WHEN 'check' THEN 1 ELSE 2 END,created LIMIT 1").fetchone()
                        job=dict(row) if row else None
                        if job:
                            db.execute("UPDATE jobs SET state='preparing',message='准备隔离环境与机器人',started=?,image_id=? WHERE id=?",(time.time(),self.node["image_id"],job["id"]))
                            if job["kind"]=="check":
                                db.execute("UPDATE submissions SET state='checking' WHERE id=?",(job["submission_a"],))
                    if job:
                        self.run(job,self.node["image_id"])
                        continue
                else:
                    self.store.execute("UPDATE jobs SET state='waiting_node',message=? WHERE state='queued'",("正在部署运行环境，就绪后自动继续" if self.deploying.is_set() else self.node["message"],))
            except Exception:
                # Server-side failures are private, and cannot silently launch a
                # trusted-process fallback. The queue remains usable afterwards.
                error=traceback.format_exc(limit=5)
                try:
                    with (self.settings.data/"worker-errors.log").open("a",encoding="utf-8") as log:
                        log.write(error[-8000:]+"\n")
                except OSError:
                    pass
            self.wake.wait(2);self.wake.clear()

    def cancelled(self,job_id):
        row=self.store.one("SELECT state FROM jobs WHERE id=?",(job_id,))
        return self.stop.is_set() or not row or row["state"]=="cancelled"

    def run(self,job,image_id):
        from embodied_arena.config import MatchConfig,ROOT
        from embodied_arena.engine import Arena
        from embodied_arena.display import composite
        arena=pool=None
        folder=self.settings.data/"jobs"/job["id"]
        folder.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        try:
            config=replace(MatchConfig.load(),scene=job["scene"],duration=job["duration"],seed=job["seed"])
            arena=Arena(config,cameras=True)
            if self.cancelled(job["id"]):
                return
            public=asdict(config);public.pop("seed",None)
            submissions=[self.store.one("SELECT * FROM submissions WHERE id=?",(job[key],)) for key in ("submission_a","submission_b")]
            orders=[validate_manifest(json.loads((self.settings.data/"submissions"/s["id"]/"submission.json").read_text(encoding="utf-8-sig"))) for s in submissions]
            pool=IsolatedPool([self.settings.data/"submissions"/s["id"] for s in submissions],image_id,self.settings,
                              self.store.setting("installation"),job["id"],public,[r.home for r in arena.robots],cancelled=lambda:self.cancelled(job["id"]),joint_orders=orders)
            if job["kind"]=="check":
                for _ in range(3):
                    pool.act([arena.observation(p) for p in (0,1)],strict=True)
                submission=submissions[0]
                reports=json.loads(submission["report"])
                reports[-1]={"name":"编译与隔离运行接口","state":"passed","detail":"包内全部 Python 文件编译成功；两个独立实例导入成功并连续 3 次返回有效动作；不代表策略能完成任务"}
                followup=None
                with self.store.connect(write=True) as db:
                    current=db.execute("SELECT state FROM jobs WHERE id=?",(job["id"],)).fetchone()
                    if current[0]=="cancelled":
                        return
                    if job.get("auto_practice"):
                        paused=db.execute("SELECT value FROM settings WHERE key='practice_paused'").fetchone()[0]=="1"
                        active=db.execute("SELECT active FROM users WHERE id=?",(submission["user_id"],)).fetchone()[0]
                        count=db.execute("SELECT count(*) FROM jobs WHERE user_id=? AND kind='practice' AND state IN ('queued','waiting_node','preparing','running')",(submission["user_id"],)).fetchone()[0]
                        if active and not paused and count<2:
                            followup=uuid.uuid4().hex
                            db.execute("INSERT INTO jobs (id,kind,user_id,submission_a,submission_b,state,message,duration,scene,seed,created) VALUES (?,'practice',?,?,?,'queued',?,?,?,?,?)",
                                       (followup,submission["user_id"],submission["id"],submission["id"],"接口检查通过，已自动加入练习队列",self.settings.practice_duration,"cyber_city",secrets.randbelow(2**31),time.time()))
                            reports[-1]["detail"]+="；已自动加入城市练习队列"
                        else:
                            reports[-1]["detail"]+="；自动练习暂未启动，请在队列恢复或空位后手动发起"
                    db.execute("UPDATE submissions SET state='ready',report=? WHERE id=?",(json.dumps(reports,ensure_ascii=False),submission["id"]))
                    db.execute("UPDATE jobs SET state='completed',message='接口检查通过，可发起练习或报名',progress=1,finished=?,result=? WHERE id=?",
                               (time.time(),json.dumps({"checked_calls":3,"image_id":image_id,"followup_job_id":followup,"compiled_all_python":True}),job["id"]))
                return
            frames=[]
            frame_folder=folder/"frames";frame_folder.mkdir(exist_ok=True)
            def capture():
                image=composite(arena)
                image.save(frame_folder/f"{len(frames):04d}.jpg",quality=82)
                frames.append({"time":round(arena.time,3),"health":[p.health for p in arena.referee.players],"scores":[p.score for p in arena.referee.players]})
                temp=folder/"replay.tmp"
                temp.write_text(json.dumps({"fps":2,"frames":frames}),encoding="utf-8")
                temp.replace(folder/"replay.json")
            capture()
            self.store.update_job(job["id"],state="running",message="仿真运行中")
            last_capture=last_update=0.
            for _ in range(math.ceil(config.duration/config.control_dt)):
                if self.cancelled(job["id"]):
                    return
                if time.monotonic()-start>self.settings.wall_timeout:
                    raise TimeoutError(f"本局超过实际运行时间上限 {self.settings.wall_timeout:g} 秒")
                actions=pool.act([arena.observation(p) for p in (0,1)])
                for event in pool.events:
                    kind=event.pop("type");arena.referee.event(arena.time,kind,**event)
                pool.events.clear()
                arena.step(actions)
                if arena.time-last_capture>=.5-1e-8:
                    capture();last_capture=arena.time
                if time.monotonic()-last_update>=1:
                    last_update=time.monotonic()
                    self.store.update_job(job["id"],progress=min(1.,arena.time/config.duration),message=f"仿真 {arena.time:.1f} / {config.duration:g} 秒")
            if frames[-1]["time"]<arena.time-1e-8:
                capture()
            result=arena.save(folder)
            platform_hash=hashlib.sha256()
            for source in sorted((ROOT/"embodied_arena").rglob("*.py")):
                platform_hash.update(source.relative_to(ROOT).as_posix().encode())
                platform_hash.update(source.read_bytes())
            reproduction={"job_id":job["id"],"image_id":image_id,"environment":self.settings.environment,
                          "submissions":[{"id":s["id"],"sha256":s["sha256"],"user_id":s["user_id"]} for s in submissions],
                          "config":asdict(config),"wall_seconds":round(time.monotonic()-start,2),
                          "platform_version":importlib.metadata.version("embodied-arena"),
                          "platform_source_sha256":platform_hash.hexdigest(),"runner_limits":self.settings.limits(),
                          "wire_protocol":"bounded-json-array-v1"}
            (folder/"reproduction.json").write_text(json.dumps(reproduction,ensure_ascii=False,indent=2),encoding="utf-8")
            result.pop("seed",None)
            result["wall_seconds"]=reproduction["wall_seconds"]
            self.store.update_job(job["id"],state="completed",message="成绩与回放已保存",progress=1.,finished=time.time(),result=result)
        except Exception as error:
            detail=str(error)[-4000:]
            (folder/"error.txt").write_text(traceback.format_exc(limit=8)[-16000:],encoding="utf-8")
            self.store.update_job(job["id"],state="failed",message=detail,finished=time.time())
            if job["kind"]=="check":
                submission=self.store.one("SELECT report FROM submissions WHERE id=?",(job["submission_a"],))
                reports=json.loads(submission["report"])
                reports[-1]={"name":"编译与隔离运行接口","state":"failed","detail":detail}
                self.store.execute("UPDATE submissions SET state='check_failed',report=? WHERE id=?",(json.dumps(reports,ensure_ascii=False),job["submission_a"]))
        finally:
            if pool:
                for pid,agent in enumerate(pool.agents):
                    (folder/f"player-{pid}.log").write_bytes(bytes(agent.logs))
                pool.close()
            if arena:
                arena.close()
