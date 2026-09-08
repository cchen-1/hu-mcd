"""Short compute-node ViT-H and masked-ResNet resource/correctness probe."""
import json
import os
from pathlib import Path
import resource
import sys
import time


def run(config, release, output, check_numerics):
    sys.path.insert(0,str(release))
    import copy
    import random
    import numpy as np
    import torch
    import classes
    import run_smoke
    from concept_explainer import ConceptExplainer
    from utils import utils_general,utils_mcd
    from torch.utils.data import DataLoader
    random.seed(43);np.random.seed(43);torch.manual_seed(43)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    utils_general.DEVICE='cuda:0'
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.cuda.reset_peak_memory_stats()
    manifest=json.loads(Path(config['dataset_manifest']).read_text())
    # Includes the sole converted validation image, plus fixed color inputs.
    converted=[e for e in manifest['validation'] if e.get('compatibility_adjustment')]
    chosen=manifest['training'][:2]+manifest['validation'][:1]+converted
    if len(chosen)!=4: raise ValueError('Expected four explicit probe images')
    images=[classes.ImageClass(e['input_path'],300) for e in chosen]
    start=time.perf_counter()
    model_start=time.perf_counter()
    sam=utils_general.load_sam_mask_generator(sam_type='vit_h',sam_checkpoint=config['segmentation']['checkpoint'],
                                             points_per_side=32,min_mask_region_area=256)
    torch.cuda.synchronize();sam_load=time.perf_counter()-model_start
    segmentation=[]
    for image in images:
        t=time.perf_counter()
        image.load_segments(cache_dir=str(Path(output)/'segments'),sam_model=sam)
        torch.cuda.synchronize()
        segmentation.append({'input':image.filename,'seconds':time.perf_counter()-t,'segments':len(image.segments)})
    del sam
    import gc
    gc.collect();torch.cuda.empty_cache()
    explainer=ConceptExplainer(config['source_dir'],'golden_retriever','resnet50','global_pool')
    explainer.segm_algo='sam'
    t=time.perf_counter()
    explainer._load_or_calc_acts(str(Path(output)/'activations'),'reference_probe',images,0,True,-1,.25,
                                config['batch_size'],False,True)
    torch.cuda.synchronize();activation_seconds=time.perf_counter()-t
    segs=[s for im in images for s in im.segments]
    acts=np.stack([s.model_act for s in segs]);logits=np.stack([s.model_pred for s in segs])
    weights=explainer.model.fc.weight.detach().cpu().numpy();bias=explainer.model.fc.bias.detach().cpu().numpy()
    fc,_=check_numerics(acts,logits,weights,bias,None)
    # End-to-end all-one masking must agree with the corresponding unmasked tensor.
    artificial=copy.copy(images[0])
    artificial.segments=[classes.SegmentClass(np.ones(artificial.img_numpy.shape[:2],dtype=np.float32),artificial),
                         classes.SegmentClass(np.zeros(artificial.img_numpy.shape[:2],dtype=np.float32),artificial)]
    ds=classes.ConceptDatasetClass([artificial],explainer.model.default_cfg,0,True,-1,1.0)
    batch=utils_general.custom_collate([ds[0],ds[1]])
    with torch.no_grad():
        masked=explainer.model(batch)
        ordinary=explainer.model(batch[0][:1])
    if not torch.isfinite(masked).all(): raise ValueError('Non-finite all-one/zero mask boundary output')
    np.testing.assert_allclose(masked[:1].cpu().numpy(),ordinary.cpu().numpy(),rtol=1e-4,atol=1e-4)
    # Independent nonorthogonal algebra fixture exercises complement and signed relevance.
    fixture=np.array([[1.,2.,3.],[-4.,1.,0.]])
    weight=np.array([2.,-3.,1.]);fixture_bias=4.
    bases=[np.array([[1.,0.,0.]]),np.array([[.6,.8,0.]]),np.array([[0.,0.,1.]])]
    algebra,_=check_numerics(fixture,fixture@weight+fixture_bias,weight,fixture_bias,bases)
    # Bounded SSC primitive timing only; do not fit concepts on this tiny sample.
    small=acts[:min(128,len(acts))]
    small=small/np.linalg.norm(small,axis=1,keepdims=True)
    t=time.perf_counter();sparse=utils_mcd.compute_sparse_repr_matrix(small,n_jobs=1)
    sparse_seconds=time.perf_counter()-t
    gpu=torch.cuda.get_device_properties(0)
    report={'status':'PASS','scope':'resource and correctness probe; no concept discovery or paper metrics',
            'inputs':[{'source':e['source'],'input_path':e['input_path'],'input_sha256':e['input_sha256']} for e in chosen],
            'sam_model':'vit_h','sam_points_per_side':32,'sam_load_seconds':sam_load,
            'segmentation':segmentation,'activation_seconds':activation_seconds,'batch_size':config['batch_size'],
            'segment_count':len(acts),'all_classes_fc':fc,'mask_boundary_finite':True,
            'all_one_mask_matches_unmasked':True,'synthetic_decomposition':algebra,
            'ssc_primitive_rows':len(small),'ssc_primitive_seconds':sparse_seconds,'ssc_nonzeros':int(sparse.nnz),
            'gpu':gpu.name,'gpu_total_mib':gpu.total_memory/1024**2,
            'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'total_seconds':time.perf_counter()-start,
            'limitation':'Four images do not bound full-dataset segment count, SSC cost, or fitted subspace reconstruction; actual fitted checks run in reference job.'}
    np.savez_compressed(Path(output)/'probe_features.npz',features=acts,logits=logits)
    (Path(output)/'probe_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
