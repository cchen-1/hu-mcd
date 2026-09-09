"""Targeted fault diagnosis for probe 28204575; no SAM or concept fitting."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def differences(actual, reference):
    import numpy as np
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    delta=np.abs(a-b);tol=1e-4+1e-4*np.abs(b)
    return {'elements':int(a.size),'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
            'rmse':float(np.sqrt(np.mean(delta**2))),
            'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
            'max_relative':float((delta/np.maximum(np.abs(b),1e-30)).max()),
            'outside_original_tolerance':int((delta>tol).sum()),
            'passes_original_tolerance':bool(np.all(delta<=tol)),
            'exact_equal':bool(np.array_equal(actual,reference)),
            'finite':bool(np.isfinite(a).all() and np.isfinite(b).all())}


def run(release, output):
    import numpy as np
    import torch
    import timm
    import random
    random.seed(43);np.random.seed(43);torch.manual_seed(43)
    from PIL import Image
    import classes
    from utils import utils_general
    job=os.environ['SLURM_JOB_ID']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result={'diagnosis_job_id':job,'host':socket.gethostname(),'status':'STARTED',
            'scope':'Single known failing input; no SAM, new sampling, concept fitting, or formal experiment',
            'research_commit':subprocess.check_output(['git','-C',str(release),'rev-parse','HEAD'],text=True).strip(),
            'started_unix':time.time(),
            'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'timm':timm.__version__,'numpy':np.__version__},
            'comparisons':{},'trace_comparisons':{}}
    def save(): (output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        defaults={'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                  'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
                  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                  'float32_matmul_precision':torch.get_float32_matmul_precision()}
        result['default_precision']=defaults
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        result['cpu_threads']=torch.get_num_threads()
        manifest_path=Path('/scratch/user/uqcche38/hu-mcd/data/reference-golden-seed43-compat-v1/dataset_manifest.json')
        if sha(manifest_path)!='dafeafcaa64d2379a236500d443e18f8a27520b9a0ad0288e79bdab7c683cd74':raise ValueError('Dataset identity changed')
        manifest=json.loads(manifest_path.read_text());first=manifest['training'][0]
        image_path=Path(first['input_path'])
        if sha(image_path)!=first['input_sha256']:raise ValueError('Failing source image changed')
        result['image']={**first,'split':'training','same_image_as_failed_boundary_check':True}
        with Image.open(image_path) as im:result['image']['source_mode']=im.mode
        image=classes.ImageClass(str(image_path),300)
        image.segments=[classes.SegmentClass(np.ones(image.img_numpy.shape[:2],dtype=np.float32),image),
                        classes.SegmentClass(np.zeros(image.img_numpy.shape[:2],dtype=np.float32),image)]
        utils_general.DEVICE='cpu'
        model=utils_general.make_model('resnet50')
        weight_path=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/resnet50_a1_0-14fe96d1.pth'
        if sha(weight_path)!='14fe96d1f9fb311a60490082d2077e6e60427dcfe21839ddf934cce948f72b0f':raise ValueError('Weights changed')
        result['classifier_sha256']=sha(weight_path)
        model_path=Path(timm.__file__).parent/'models'
        result['source_identity']={}
        for name in ('resnet.py','sal_layers.py'):
            installed=sha(model_path/name);expected=sha(Path(release)/'input_masking'/name)
            result['source_identity'][name]={'installed':installed,'release':expected,'identical':installed==expected}
            if installed!=expected:raise ValueError('Installed masking implementation changed')
        result['model_training']=model.training
        result['training_batchnorm_modules']=[name for name,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d) and m.training]
        ds=classes.ConceptDatasetClass([image],model.default_cfg,0,True,-1,1.0)
        x=ds[0][0].unsqueeze(0);one=ds[0][1].unsqueeze(0);zero=ds[1][1].unsqueeze(0)
        if not torch.equal(ds[0][0],ds[1][0]) or not torch.all(one==1) or not torch.all(zero==0):raise ValueError('Inputs are not the intended identical image and binary boundary masks')
        result['input_tensor']={'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                                'same_image_for_all_comparisons':True,'one_mask_exactly_one':True,'zero_mask_exactly_zero':True}
        def model_digest():
            digest=hashlib.sha256()
            for name,value in model.state_dict().items():
                digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        result['state_sha256_before']=model_digest()
        cache=Path('/scratch/user/uqcche38/hu-mcd/launches/28204575/probe/activations')
        stem='reference_probe_resnet50_global_pool_sam_org_scale_masked_conv_leakage_erosion_0.25_5d6b64192ebc92ef1a5e0d56eef617c079d1f0b1598bb0183fbca1d2c47eccf0'
        acts_path=cache/(stem+'_acts.npy');logits_path=cache/(stem+'_logits.npy')
        acts=np.load(acts_path,allow_pickle=False);cached_logits=np.load(logits_path,allow_pickle=False)
        reconstructed=acts @ model.fc.weight.detach().numpy().T + model.fc.bias.detach().numpy()
        result['original_cached_fc_reconstruction']={**differences(reconstructed,cached_logits),
            'feature_shape':list(acts.shape),'logit_shape':list(cached_logits.shape),
            'feature_sha256':sha(acts_path),'logit_sha256':sha(logits_path),
            'zero_feature_rows':int(np.all(acts==0,axis=1).sum())}
        arrays={};traces={}
        trace_names=['conv1','bn1','act1','maxpool']+[f'layer{n}' for n in range(1,5)]+['global_pool','fc']
        def forward(label, device, n, masked, mixed=False, trace=False):
            xx=x.to(device).repeat(n,1,1,1)
            mm=one.to(device).repeat(n,1,1,1)
            if mixed and n>1:mm[-1]=0
            hooks=[];values={}
            if trace:
                for name in trace_names:
                    def hook(_m,_args,out,name=name):
                        a=out[0] if isinstance(out,tuple) else out
                        values[name]=a[:1].detach().cpu().numpy().copy()
                    hooks.append(model.get_submodule(name).register_forward_hook(hook))
            try:
                with torch.no_grad():y=model((xx,mm,-1) if masked else xx)
                arrays[label]=y[:1].detach().cpu().numpy().copy()
                if not torch.isfinite(y).all():raise ValueError('Non-finite outputs: '+label)
            finally:
                for h in hooks:h.remove()
            if trace:traces[label]=values
            return y
        def compare(label,a,b):result['comparisons'][label]=differences(arrays[a],arrays[b])
        def phase(prefix,device,extra=False):
            model.to(device)
            forward(prefix+'_plain1',device,1,False,trace=True)
            forward(prefix+'_plain2',device,2,False,trace=True)
            forward(prefix+'_mixed2',device,2,True,True,trace=True)
            forward(prefix+'_masked1',device,1,True)
            forward(prefix+'_ones2',device,2,True)
            forward(prefix+'_plain1_repeat',device,1,False)
            compare(prefix+'_original_mixed2_vs_plain1',prefix+'_mixed2',prefix+'_plain1')
            compare(prefix+'_plain2_vs_plain1',prefix+'_plain2',prefix+'_plain1')
            compare(prefix+'_same_batch2_masked_vs_plain',prefix+'_mixed2',prefix+'_plain2')
            compare(prefix+'_same_batch1_masked_vs_plain',prefix+'_masked1',prefix+'_plain1')
            compare(prefix+'_zero_neighbor_vs_one_neighbor',prefix+'_mixed2',prefix+'_ones2')
            compare(prefix+'_repeat_same_forward',prefix+'_plain1_repeat',prefix+'_plain1')
            for pair,a,b in [('batch_size','plain2','plain1'),('mask_branch','mixed2','plain2')]:
                result['trace_comparisons'][prefix+'_'+pair]={name:differences(traces[prefix+'_'+a][name],traces[prefix+'_'+b][name]) for name in trace_names}
            if extra:
                forward(prefix+'_plain8',device,8,False)
                forward(prefix+'_mixed8',device,8,True,True)
                compare(prefix+'_same_batch8_masked_vs_plain',prefix+'_mixed8',prefix+'_plain8')
                compare(prefix+'_plain8_vs_plain1',prefix+'_plain8',prefix+'_plain1')
            save()
        phase('cpu','cpu')
        if torch.cuda.is_available():
            result['gpu_name']=torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
            phase('cuda_default','cuda',True)
            # Change exactly convolution TF32; leave all other settings untouched.
            torch.backends.cudnn.allow_tf32=False
            phase('cuda_cudnn_tf32_off','cuda',True)
            torch.backends.cudnn.allow_tf32=defaults['cudnn_allow_tf32']
            forward('cuda_restored_mixed2','cuda',2,True,True)
            forward('cuda_restored_plain1','cuda',1,False)
            compare('cuda_restored_original','cuda_restored_mixed2','cuda_restored_plain1')
            compare('cuda_default_vs_restored','cuda_restored_mixed2','cuda_default_mixed2')
            compare('cuda_default_same_batch_precision_effect','cuda_default_plain2','cuda_cudnn_tf32_off_plain2')
            result['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        else:result['gpu_note']='No GPU in this allocation; CUDA-specific root cause remains untested'
        result['state_sha256_after']=model_digest()
        result['model_state_unchanged']=result['state_sha256_before']==result['state_sha256_after']
        if not result['model_state_unchanged']:raise ValueError('Model state changed during diagnosis')
        result['class_index']=int(utils_general.get_imagenet_class_index('golden_retriever',str(Path(release)/'imagenet1k_class_info.json')))
        for label,vals in arrays.items():
            result.setdefault('predictions',{})[label]={'argmax':int(vals[0].argmax()),'golden_retriever_logit':float(vals[0,result['class_index']])}
        np.savez_compressed(output/'logits.npz',**arrays)
        flat={label+'__'+name:arr for label,values in traces.items() for name,arr in values.items()}
        np.savez_compressed(output/'layer_traces.npz',**flat)
        np.save(output/'input_tensor.npy',x.numpy(),allow_pickle=False)
        result['artifacts']={name:{'bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name in ('logits.npz','layer_traces.npz','input_tensor.npy')}
        result['status']='DIAGNOSIS_COMPLETED'
    except BaseException as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['elapsed_seconds']=time.time()-result['started_unix']
        save()
    print(json.dumps({'status':result['status'],'output':str(output),'comparisons':result['comparisons']},indent=2))


if __name__=='__main__':
    import re
    if not os.environ.get('SLURM_JOB_ID','').isdigit() or not re.fullmatch(r'bun[0-9]{3}',socket.gethostname().split('.')[0]):
        raise RuntimeError('Slurm compute node required')
    release=Path(sys.argv[1]);sys.path.insert(0,str(release))
    run(release,Path(sys.argv[2]))
