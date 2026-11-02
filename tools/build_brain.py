"""Build a small fsaverage ico5 pial mesh + region labels for the aboutme glass brain."""
import base64, json, sys
import numpy as np, nibabel as nib
from mne.surface import _get_ico_surface

SUBJ = "/home/co/.cache/neuralset/MNE-sample-data/subjects/fsaverage"
N = 10242                      # ico5: the first 10242 fsaverage vertices
tris = _get_ico_surface(5)["tris"].astype(np.uint16)
REGIONS = {  # region -> group id used by the shader (0 = none)
    "lateraloccipital": 1, "pericalcarine": 1, "cuneus": 1, "lingual": 1,
    "superiortemporal": 2, "transversetemporal": 2, "bankssts": 2, "middletemporal": 2,
    "parsopercularis": 3, "parstriangularis": 3,
}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
out = {"n": N, "hemis": []}
allpos = []
for h in ("lh", "rh"):
    pos, _ = nib.freesurfer.read_geometry(f"{SUBJ}/surf/{h}.pial")
    allpos.append(pos[:N])
c = np.concatenate(allpos).mean(0); scale = np.abs(np.concatenate(allpos) - c).max()
for h, pos in zip(("lh", "rh"), allpos):
    lab, _, names = nib.freesurfer.read_annot(f"{SUBJ}/label/{h}.aparc.annot")
    names = [n.decode() for n in names]
    grp = np.array([REGIONS.get(names[l], 0) if l >= 0 else 0 for l in lab[:N]], np.uint8)
    sulc = nib.freesurfer.read_morph_data(f"{SUBJ}/surf/{h}.sulc")[:N]
    sulc = np.clip((sulc - sulc.min()) / (np.ptp(sulc)) * 255, 0, 255).astype(np.uint8)
    q = np.round((pos - c) / scale * 32000).astype(np.int16)
    out["hemis"].append({"h": h, "pos": b64(q), "grp": b64(grp), "sulc": b64(sulc), "tri": b64(tris)})
open(sys.argv[1], "w").write("window.BRAIN=" + json.dumps(out) + ";\n")
