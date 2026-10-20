"""Evaluate photometric optimisation on synthetic datasets."""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from panoforge.align import create_project, align_images
from panoforge.photometric import optimize_photometric
for d, mode in [('testdata/rect_row','auto'), ('testdata/rect_jitter','full')]:
    truth=json.load(open(d+'/truth.json'))
    p=create_project([d+'/'+t['file'] for t in truth['images']])
    p.optimizer.distortion='heavy_shift'
    align_images(p, photometric=False)
    p.exposure.optimize_exposure=mode; p.exposure.optimize_wb='full' if mode=='full' else 'auto'
    r=optimize_photometric(p)
    print(d, {k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
    print(' vig', np.round(p.lenses[0].vig,3), 'true', truth['vignetting'])
    tev=np.array([t['ev'] for t in truth['images']]); eev=np.array([im.ev for im in p.images])
    print(' ev err', np.round((eev-eev[0])-(tev-tev[0]),3))
    twb=np.array([t['wb'] for t in truth['images']]); ewb=np.array([[im.wb_r,im.wb_b] for im in p.images])
    print(' wb err (max)', np.round(np.abs(ewb/ewb[0]-twb/twb[0]).max(),4))
    p.save(d+'/aligned.pfproj')
