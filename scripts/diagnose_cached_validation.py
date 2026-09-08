"""Diagnose existing HU-MCD caches locally; no inference, cache mutation or SSH."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, csv, hashlib, json, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse, linalg
from scipy.optimize import linear_sum_assignment
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import classes
from utils import utils_mcd
torch.set_num_threads(4)

class SegmentRecord:
    def __init__(self, model_act, org_img):
        self.model_act = model_act
        self.org_img = org_img

def one(directory, pattern):
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]

def score(x, bases):
    # Complement is perpendicular to the learned union. Solve the small learned
    # system, then obtain complement from the residual. Validate against upstream.
    b = np.concatenate(bases).astype(np.float64)
    x = x.astype(np.float64)
    coeff = linalg.lstsq(b.T, x.T)[0]
    cut = np.cumsum([0] + [len(basis) for basis in bases])
    parts = [coeff[s:e].T @ b[s:e] for s,e in zip(cut[:-1],cut[1:])]
    residual = x - coeff.T @ b
    out = np.stack([linalg.norm(p,axis=1) for p in parts+[residual]],axis=1)
    out /= linalg.norm(x,axis=1)[:,None]
    union_fraction = 1-out[:,-1]**2
    return out, union_fraction

def stats(x):
    return {"min":float(np.min(x)), "median":float(np.median(x)),
            "max":float(np.max(x)), "mean":float(np.mean(x))}

def run(name, summary_path, cfg_path, train_path, val_path, sparse_path, masks_root):
    summary = json.loads(summary_path.read_text())
    cfg = json.loads(cfg_path.read_text())
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    raw = {"training":np.load(train_path,allow_pickle=False),
           "validation":np.load(val_path,allow_pickle=False)}
    images={}; filtered={}; records={}; quality={}; raw_counts={}
    for split,key,folder in [("training","train","segments_train"),
                             ("validation","validation","segments_validation")]:
        masks = sorted((masks_root/folder).glob("*_sam.npy"))[:summary[key+"_images"]]
        assert len(masks)==summary[key+"_images"]
        counts=[len(np.load(p,allow_pickle=False,mmap_mode="r")) for p in masks]
        raw_counts[split]=counts
        assert sum(counts)==len(raw[split]), (name,split,counts,raw[split].shape)
        assert np.isfinite(raw[split]).all()
        images[split]=[]; records[split]=[]; offset=0
        for idx,(path,n) in enumerate(zip(masks,counts)):
            img=SimpleNamespace(filename=path.name.removesuffix("_sam.npy"), segments=[])
            for j,act in enumerate(raw[split][offset:offset+n]):
                if np.linalg.norm(act)==0: continue
                img.segments.append(SegmentRecord(model_act=act,org_img=img))
                records[split].append({"image_index":idx+1,"filename":img.filename,
                                       "raw_segment_index":j})
            images[split].append(img); offset+=n
        assert [len(i.segments) for i in images[split]]==summary[key+"_segments_per_image"]
        filtered[split]=np.stack([s.model_act for im in images[split] for s in im.segments])
        quality[split]={"raw_shape":list(raw[split].shape), "all_finite":True,
                        "removed_zero_rows":len(raw[split])-len(filtered[split]),
                        "nonzero_norm":stats(np.linalg.norm(filtered[split],axis=1))}
    space=classes.ClusterSpaceClass(images["training"],norm_acts=True)
    space.sparse_repr_matrix=sparse.load_npz(sparse_path)
    space.outlier_mask=np.load(str(sparse_path).replace(".npz","_outlier.npy"),allow_pickle=False)
    assert space.sparse_repr_matrix.shape == (len(filtered["training"]),)*2
    assert np.isfinite(space.sparse_repr_matrix.data).all()
    space.sparse_subspace_clustering(str(sparse_path),cfg["clustering"]["outlier_percentile"],
                                    cfg["clustering"]["n_clusters"],None)
    concepts=[classes.ConceptClass(c.label,c.segments,c.norm_acts,None)
              for c in space.clusters if len(c.segments)>=cfg["clustering"]["min_cluster_size"]]
    for c in concepts:
        c.compute_pca_basis(len(concepts),cfg["clustering"]["subspace_dimensionality"],"ratio")
    bases=[c.basis for c in concepts]
    dims=[len(b) for b in bases]
    assert dims==summary["concept_dimensions"], (name,dims,summary["concept_dimensions"])
    result={"name":name,"summary":str(summary_path),"config":str(cfg_path),
            "array_quality":quality,"dimensions":dims,
            "union_rank":int(np.linalg.matrix_rank(np.concatenate(bases))),
            "union_condition":float(np.linalg.cond(np.concatenate(bases))),
            "clusters":[],"splits":{}}
    for c,b in zip(concepts,bases):
        x=np.stack([s.model_act for s in c.segments])
        centroid=x.mean(axis=0)
        result["clusters"].append({"label":int(c.label),"segments":len(x),
            "distinct_images":len(set(s.org_img.filename for s in c.segments)),
            "dimension":len(b),
            "centroid_energy_in_own_basis":float(np.linalg.norm(b@centroid)**2/np.linalg.norm(centroid)**2)})
    scores={}
    for split,x in filtered.items():
        scores[split],energy=score(x,bases)
        winners=scores[split].argmax(axis=1)
        counts=[int((winners==i).sum()) for i in range(len(bases))]
        expected=summary["prototype_quality_proxy"][split]["concept_assignment_counts"]
        assert counts==expected, (name,split,counts,expected)
        for record,row,frac in zip(records[split],scores[split],energy):
            record.update(winner=int(row.argmax()),best_learned=float(row[:-1].max()),
                          complement=float(row[-1]),union_energy=float(frac))
        result["splits"][split]={"learned_counts":counts,
            "learned_total":sum(counts),"segments":len(x),
            "best_learned":stats(scores[split][:,:-1].max(axis=1)),
            "complement":stats(scores[split][:,-1]),"union_energy":stats(energy),
            "per_image":[{"image_index":i+1,"filename":im.filename,
                          "segments":len(im.segments),
                          "learned":sum(r["winner"]<len(bases) for r in records[split] if r["image_index"]==i+1)}
                         for i,im in enumerate(images[split])]}
    # Check the algebraic shortcut against unmodified upstream full decomposition.
    complement=utils_mcd.basis_of_ortho_complement(bases)
    check=utils_mcd.batch_concept_activations(
        filtered["validation"][:3],bases+[complement],False)
    error=float(np.max(np.abs(check-scores["validation"][:3])))
    assert error < 2e-5, error
    result["projection_crosscheck_max_abs_error"]=error
    result["reproduced_saved_dimensions_and_assignments"]=True
    result["inputs_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [summary_path,cfg_path,train_path,val_path,sparse_path,
                                      Path(str(sparse_path).replace(".npz","_outlier.npy"))]}
    for split in records:
        with (OUT/(name+"_"+split+"_segments.csv")).open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(records[split][0]));w.writeheader();w.writerows(records[split])
    np.savez(OUT/(name+"_reconstructed.npz"),train_scores=scores["training"],
             validation_scores=scores["validation"],labels=space.labels,
             **{f"basis_{i}":b for i,b in enumerate(bases)})
    print(json.dumps(result,indent=2),flush=True)
    return result,raw,raw_counts,bases,filtered

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wsl-root",type=Path,default=Path("/home/chen/results/hu-mcd-fidelity-imagewoof320"))
parser.add_argument("--bunya-cache",type=Path,default=REPO/"artifacts/bunya/validation-diagnosis-2026-09-08/cache")
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
OUT=args.output; OUT.mkdir(parents=True,exist_ok=False)
wroot=args.wsl_root
w=run("wsl20",wroot/"output/summary.json",REPO/"configs/fidelity_imagewoof320.json",
      one(wroot/"cache/activations_train","*_acts.npy"),
      one(wroot/"cache/activations_validation","*_acts.npy"),
      wroot/"cache/self_representation/sparse_repr_matrix_golden_retriever_20.npz",wroot/"cache")
b=run("bunya10",REPO/"outputs/bunya-phase3-smoke/summary.json",
      REPO/"hpc/configs/phase3_smoke_bunya.json",args.bunya_cache/"train_acts.npy",
      args.bunya_cache/"validation_acts.npy",
      args.bunya_cache/"sparse_repr_matrix_golden_retriever_10.npz",wroot/"cache")
comparison={}
for split in ["training","validation"]:
    wa=w[1][split];ba=b[1][split];offset=0;matches=[]
    for count in b[2][split]:
        a=wa[offset:offset+count].astype(float);c=ba[offset:offset+count].astype(float)
        # Segment labels are random; align by minimum Euclidean distance within each image.
        cost=linalg.norm(a[:,None,:]-c[None,:,:],axis=2)
        ix,jx=linear_sum_assignment(cost)
        for i,j in zip(ix,jx):
            na=np.linalg.norm(a[i]);nc=np.linalg.norm(c[j])
            if na and nc: matches.append((np.dot(a[i],c[j])/na/nc,cost[i,j]/na))
            else: assert na==nc==0
        offset+=count
    comparison[split]={"matched_nonzero_segments":len(matches),
                       "matched_cosine":stats(np.array(matches)[:,0]),
                       "matched_relative_l2":stats(np.array(matches)[:,1])}
frozen,energy=score(w[4]["validation"][:len(b[4]["validation"])],w[3])
comparison["wsl20_basis_first5_validation"]={
    "learned":int((frozen.argmax(axis=1)<len(w[3])).sum()),"segments":len(frozen)}
for label,x,bases in [
    ("wsl20_basis_bunya_validation",b[4]["validation"],w[3]),
    ("bunya10_basis_wsl_first5_validation",w[4]["validation"][:len(b[4]["validation"])],b[3]),
    ("bunya10_basis_wsl_all10_validation",w[4]["validation"],b[3])]:
    frozen,_=score(x,bases)
    comparison[label]={"learned":int((frozen.argmax(axis=1)<len(bases)).sum()),"segments":len(x)}
result={"runs":[w[0],b[0]],"comparison":comparison,
        "limitations":["Bases and labels reconstructed from saved SSC matrices using current local libraries; "
                        "saved dimensions and assignment counts are asserted, projection norms are cross-checked.",
                       "Bunya raw segment boundaries use matching WSL per-image mask counts; per-image feature matching "
                       "checks this correspondence. Remote source-image/mask byte identity is not asserted.",
                       "No new model inference, segmentation, sparse fitting, GPU work or Slurm submission."]}
(OUT/"diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")
print("COMPARISON",json.dumps(comparison,indent=2),flush=True)
