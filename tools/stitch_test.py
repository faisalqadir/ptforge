import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from panoforge.project import Project
from panoforge.stitcher import stitch
p = Project.load(sys.argv[1])
out = sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else None
for k, v in (a.split('=') for a in sys.argv[4:]):
    setattr(p.output, k, v)
t = time.time()
r = stitch(p, out, width=w, progress=lambda st, i, n, **k: None)
print(f"{r.width}x{r.height} in {r.seconds:.1f}s -> {r.path} {r.extra_files}")
