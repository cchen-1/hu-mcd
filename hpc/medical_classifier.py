"""Proposed M single-fit source classifier. No execution without Slurm + approval.

The complete epoch budget finishes before selection is sealed and test images are
decoded. Checkpoints and all selection/identity evidence survive failed checks.
This is a new medical path, not a claim of GPU validation or convergence.
"""
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from hpc.medical_protocol import (execution_gate, bound_inputs, manifest_rows,
                                  learning_rate, improves, require_frozen_test, digest)
from utils.run_tracking import atomic_json, sha256, utc_now


class SourceImages(Dataset):
    def __init__(self, images, labels, rows, protocol, training=False):
        if (images.shape != (len(rows),224,224,3) or images.dtype != np.uint8 or
                not np.array_equal(labels.reshape(-1), [int(r['label']) for r in rows])):
            raise ValueError('Array/manifest geometry, labels or row mapping differ')
        self.images, self.labels, self.rows = images, labels.reshape(-1), rows
        self.mean = torch.tensor(protocol['mean'], dtype=torch.float32)[:,None,None]
        self.std = torch.tensor(protocol['std'], dtype=torch.float32)[:,None,None]
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i].copy()).permute(2,0,1).float().div_(255)
        if self.training:
            if torch.rand(()) < .5: x = x.flip(2)
            if torch.rand(()) < .5: x = x.flip(1)
        return (x-self.mean)/self.std, int(self.labels[i]), i


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def metrics(labels, logits):
    from scipy.special import softmax
    from sklearn.metrics import confusion_matrix, roc_auc_score
    labels, logits = np.asarray(labels), np.asarray(logits)
    if logits.shape != (len(labels),7) or not np.isfinite(logits).all():
        raise ValueError('Nonfinite or incorrectly shaped classifier output')
    counts = np.bincount(labels, minlength=7)
    if len(counts) != 7 or np.any(counts == 0):
        raise ValueError('Undefined seven-class metric; no replacement selection criterion')
    probs = softmax(logits, axis=1); pred = logits.argmax(axis=1)
    cm = confusion_matrix(labels, pred, labels=list(range(7)))
    recalls = np.diag(cm)/counts
    aucs = [roc_auc_score(labels == i, probs[:,i]) for i in range(7)]
    return dict(accuracy=float(np.mean(pred == labels)), balanced_accuracy=float(recalls.mean()),
                macro_ovr_auc=float(np.mean(aucs)), per_class_auc=aucs,
                per_class_recall=recalls.tolist(), positives=counts.tolist(),
                negatives=(len(labels)-counts).tolist(), confusion_matrix=cm.tolist(),
                predicted_class_counts=np.bincount(pred,minlength=7).tolist(),
                constant_argmax=bool(len(np.unique(pred)) == 1))


def infer(model, loader, device):
    model.eval(); predictions=[]; labels=[]; indices=[]; loss_sum=0.
    with torch.inference_mode():
        for x,y,idx in loader:
            logits = model(x.to(device)); y = y.to(device)
            loss = torch.nn.functional.cross_entropy(logits,y,reduction='sum')
            if not torch.isfinite(loss): raise ValueError('Nonfinite evaluation loss')
            predictions.append(logits.cpu().numpy()); labels.extend(y.cpu().tolist())
            indices.extend(idx.tolist()); loss_sum += float(loss)
    if indices != list(range(len(loader.dataset))):
        raise ValueError('Evaluation rows are reordered, missing or duplicated')
    logits = np.concatenate(predictions); result = metrics(labels,logits)
    result['cross_entropy'] = loss_sum/len(indices)
    return result,logits


def save_predictions(path, dataset, logits):
    np.savez_compressed(path, logits=logits, labels=dataset.labels,
                        array_row=np.arange(len(dataset)),
                        image_id=np.array([r['image_id'] for r in dataset.rows]),
                        lesion_id=np.array([r['lesion_id'] for r in dataset.rows]))


def atomic_torch(path, value):
    tmp = path.with_suffix(path.suffix+'.part'); torch.save(value,tmp); os.replace(tmp,path)


def save_state(folder, model, optimizer, generator, epoch, criterion):
    folder.mkdir(exist_ok=True)
    # Separate files keep each below the established 256-MiB collection limit.
    atomic_torch(folder/'model.pt',model.state_dict())
    atomic_torch(folder/'optimizer.pt',optimizer.state_dict())
    atomic_torch(folder/'rng.pt',dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                 train_loader=generator.get_state()))
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in folder.glob('*.pt')}
    if any(e['bytes']>256*1024**2 for e in files.values()):
        raise RuntimeError('Checkpoint component exceeds existing collector file limit')
    atomic_json(folder/'identity.json',dict(epoch=epoch,criterion=criterion,files=files))


def adapter_checks(model, x, output, rows):
    """Independent ordinary reference, same batch8, plus saved linear-head algebra."""
    from torchvision.models import resnet50
    ordinary=resnet50(weights=None,num_classes=7).to(x.device).eval()
    ordinary.load_state_dict(model.state_dict(),strict=True); model.eval()
    stored={}
    hook=ordinary.avgpool.register_forward_hook(lambda _,inp,out: stored.update(ordinary_features=out.flatten(1)))
    with torch.inference_mode():
        ordinary_logits=ordinary(x); hook.remove()
        f=model.forward_head(model.forward_features(x),pre_logits=True)
        logits=model(x)
        mf=model.forward_head(model.forward_features((x,torch.ones_like(x[:,:1]),1)),pre_logits=True)
        masked_logits=model((x,torch.ones_like(x[:,:1]),1))
        algebra=torch.nn.functional.linear(f,model.fc.weight,model.fc.bias)
    arrays={k:v.detach().cpu().numpy() for k,v in dict(ordinary_features=stored['ordinary_features'],
            features=f,masked_features=mf,ordinary_logits=ordinary_logits,logits=logits,
            masked_logits=masked_logits,head_algebra=algebra).items()}
    np.savez_compressed(Path(output)/'adapter_check_arrays.npz',**arrays,
                        image_id=np.array([r['image_id'] for r in rows]))
    checks=[]
    for a,b,tol in [('ordinary_features','features',1e-4),('features','masked_features',1e-4),
                    ('ordinary_logits','logits',1e-4),('logits','masked_logits',1e-4),
                    ('logits','head_algebra',1e-5)]:
        passed=bool(np.isfinite(arrays[a]).all() and np.isfinite(arrays[b]).all() and
                    np.allclose(arrays[a],arrays[b],rtol=tol,atol=tol))
        checks.append(dict(a=a,b=b,rtol=tol,atol=tol,pass_check=passed,
                           max_abs_difference=float(np.max(np.abs(arrays[a]-arrays[b])))))
    atomic_json(Path(output)/'adapter_checks.json',dict(checks=checks,batch_size=len(x),
                selection='First eight fixed validation rows in accepted array order',
                independent_reference='torchvision ordinary ResNet50; strict identical state'))
    if not all(c['pass_check'] for c in checks):
        raise ValueError('Same-batch ordinary/masked/head equivalence failed; evidence retained')
    del ordinary


def run(config, output):
    execution_gate(config,'medical-classifier')
    p=config['protocol']; out=Path(output); start=time.monotonic()
    atomic_json(out/'medical_progress.json',dict(stage='M_input_binding',utc=utc_now()))
    bound_inputs(p,out)
    import timm
    import timm.models.resnet as installed_resnet
    from hpc.workstream_runtime import event
    if timm.__version__!='0.6.13' or not torch.cuda.is_available():
        raise RuntimeError('Pinned timm0.6.13 and GPU required; no fallback')
    release=Path(__file__).resolve().parents[1]
    installed=Path(installed_resnet.__file__).parent
    for name in ('resnet.py','sal_layers.py'):
        if sha256(installed/name)!=sha256(release/'input_masking'/name):
            raise ValueError('Installed HU-compatible model code differs from pinned release')
    torch.set_num_threads(min(4,int(os.environ['SLURM_CPUS_PER_TASK'])))
    precision=p['precision']
    torch.set_float32_matmul_precision(precision['float32_matmul_precision'])
    torch.backends.cudnn.allow_tf32=precision['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=precision['matmul_allow_tf32']
    torch.backends.cudnn.benchmark=precision['cudnn_benchmark']
    torch.backends.cudnn.deterministic=precision['cudnn_deterministic']
    torch.use_deterministic_algorithms(precision['deterministic_algorithms'])
    seed=p['classifier_training_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    atomic_json(out/'medical_environment.json',dict(torch=torch.__version__,timm=timm.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
                precision=precision,classifier_training_seed=seed,sampling_rng_used=False))
    rows=manifest_rows(p['inputs']['source_manifest']['path'])
    shutil.copyfile(p['inputs']['source_manifest']['path'],out/'input_manifest.csv')
    datasets={}
    # Test image array is deliberately not decoded here. Metadata is not test outcome access.
    with np.load(p['inputs']['source_npz']['path'],allow_pickle=False) as z:
        for split in ('train','val'):
            datasets[split]=SourceImages(z[split+'_images'],z[split+'_labels'],rows[split],p,split=='train')
    generator=torch.Generator().manual_seed(seed)
    def loader(ds,train=False):
        return DataLoader(ds,batch_size=p['batch_size'],shuffle=train,drop_last=False,
                          num_workers=2,worker_init_fn=worker_seed,generator=generator if train else None,
                          persistent_workers=False,pin_memory=True)
    train_loader=loader(datasets['train'],True); val_loader=loader(datasets['val'])
    model=timm.create_model('resnet50',pretrained=False)
    model.load_state_dict(torch.load(p['inputs']['initializer']['path'],map_location='cpu'),strict=True)
    model.reset_classifier(7); model.to('cuda:0')
    if model.fc.in_features!=2048 or model.fc.out_features!=7 or model.fc.bias is None:
        raise ValueError('Unexpected medical head architecture')
    o=p['optimizer']; optimizer=torch.optim.Adam(model.parameters(),lr=o['lr'],betas=tuple(o['betas']),
                                               eps=o['eps'],weight_decay=o['weight_decay'])
    best=None; best_epoch=None; steps=0; visits=0; last_epoch_seconds=0.
    def budget(reserve=0):
        if time.monotonic()-start+reserve >= p['work_seconds_ceiling']:
            raise TimeoutError('M work ceiling reached/projected; no extension or automatic resume')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>p['output_bytes_ceiling']:
            raise RuntimeError('M output ceiling reached; no expansion')
    for epoch in range(1,p['epochs']+1):
        budget(last_epoch_seconds+120)
        epoch_start=time.monotonic(); lr=learning_rate(epoch,p)
        for group in optimizer.param_groups:group['lr']=lr
        model.train(); seen=[]; loss_sum=0.
        for x,y,idx in train_loader:
            x=x.to('cuda:0'); y=y.to('cuda:0')
            optimizer.zero_grad(set_to_none=True); logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if not torch.stack([torch.isfinite(param.grad).all() for param in model.parameters()
                                if param.grad is not None]).all():
                raise ValueError('Nonfinite training gradient')
            optimizer.step(); steps+=1; visits+=len(y); loss_sum+=float(loss.detach())*len(y)
            seen.extend(idx.tolist())
            if time.monotonic()-start >= p['work_seconds_ceiling']:
                raise TimeoutError('M time ceiling reached; preserve previous epoch checkpoint')
        np.save(out/f'train_order_epoch_{epoch:03d}.npy',np.asarray(seen,dtype=np.int32),allow_pickle=False)
        if sorted(seen)!=list(range(len(datasets['train']))):
            raise ValueError('Training epoch omitted/duplicated rows')
        vm,vl=infer(model,val_loader,'cuda:0'); score=vm['macro_ovr_auc']; selected=improves(score,best)
        save_state(out/'last',model,optimizer,generator,epoch,score)
        if selected:
            best=score;best_epoch=epoch
            save_state(out/'best',model,optimizer,generator,epoch,score)
            save_predictions(out/'best_validation.npz',datasets['val'],vl)
        rec=dict(epoch=epoch,lr=lr,training_cross_entropy=loss_sum/len(seen),validation=vm,
                 selected=selected,best_epoch=best_epoch,best_score=best,optimizer_steps=steps,image_visits=visits)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
        atomic_json(out/'medical_progress.json',dict(stage='M_training',completed_epochs=epoch,
                    planned_epochs=p['epochs'],**{k:rec[k] for k in ('best_epoch','best_score','optimizer_steps','image_visits')},utc=utc_now()))
        if vm['constant_argmax']:
            event(out,'M_training','MED26-M-CONSTANT',dict(epoch=epoch,predicted_class_counts=vm['predicted_class_counts']),
                  'Observed argmax collapse; not alone proof of a code defect','Record trajectory; selected-final collapse requires review',status='REVIEW_REQUIRED')
        last_epoch_seconds=time.monotonic()-epoch_start; budget()
    if steps!=p['workload']['optimizer_updates'] or visits!=p['workload']['training_image_visits']:
        raise ValueError('Planned training workload was not completed')
    checkpoint=out/'best/model.pt'
    frozen=dict(checkpoint_sha256=sha256(checkpoint),selected_epoch=best_epoch,criterion=best,
                test_outcomes_used_for_selection=False,protocol_sha256=digest(p),frozen_at_utc=utc_now())
    atomic_json(out/'frozen_classifier.json',frozen)
    atomic_json(out/'medical_progress.json',dict(stage='M_frozen_adapter_checks',completed_epochs=p['epochs'],utc=utc_now()))
    model.load_state_dict(torch.load(checkpoint,map_location='cuda:0'),strict=True)
    x=torch.stack([datasets['val'][i][0] for i in range(8)]).to('cuda:0')
    adapter_checks(model,x,out,rows['val'][:8]); budget(120)
    require_frozen_test(frozen,sha256(checkpoint),p['epochs'],p['epochs'])
    atomic_json(out/'medical_progress.json',dict(stage='M_frozen_final_test',completed_epochs=p['epochs'],utc=utc_now()))
    with np.load(p['inputs']['source_npz']['path'],allow_pickle=False) as z:
        test=SourceImages(z['test_images'],z['test_labels'],rows['test'],p)
    tm,tl=infer(model,loader(test),'cuda:0'); save_predictions(out/'final_test.npz',test,tl)
    with np.load(out/'best_validation.npz',allow_pickle=False) as v:
        selected_metrics=metrics(v['labels'],v['logits'])
    review_required=selected_metrics['constant_argmax'] or tm['constant_argmax']
    if review_required:
        event(out,'M_final','MED26-M-CONSTANT-SELECTED',dict(validation=selected_metrics,test=tm),
              'Selected classifier argmax collapse limits downstream use; no clinical quality certification',
              'Pause downstream model acceptance for diagnosis; retain fit and test; do not refit',status='REVIEW_REQUIRED')
    budget()
    atomic_json(out/'medical_progress.json',dict(stage='M_completed_pending_acceptance',completed_epochs=p['epochs'],utc=utc_now()))
    return dict(status='COMPLETED_PENDING_REVIEW' if review_required else 'COMPLETED_PENDING_ACCEPTANCE',
                fixed_classifier=frozen,validation=selected_metrics,test=tm,epochs=p['epochs'],
                optimizer_steps=steps,training_image_visits=visits,elapsed_seconds=time.monotonic()-start,
                patient_independence='NOT_ESTABLISHED',seed_stability='NOT_TESTED',
                convergence_or_clinical_quality_certified=False,medical_discovery_executed=False)
