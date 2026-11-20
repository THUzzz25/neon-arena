import hashlib
from embodied_arena.config import ROOT


def source_digest():
    digest=hashlib.sha256()
    paths=list((ROOT/'embodied_arena').rglob('*.py'))+list((ROOT/'configs').glob('*'))
    for path in sorted(p for p in paths if p.is_file()):
        digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(path.read_bytes())
    return digest.hexdigest()
