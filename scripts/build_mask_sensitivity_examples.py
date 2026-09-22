"""Private static A examples, fixed random and explicitly diagnostic selection."""
import hashlib,html,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def build(root,out,bundle):
    selections=json.loads((out/'A-display-selection.json').read_text());rows=[];body='<h2>可追溯mask变化 / Traceable mask changes</h2><p>每类从全部原始区域以独立seed20260921随机抽2例，再展示半径1侵蚀下原始行顺序中首个有效分配切换例。随机例不剔除零特征，诊断例不是总体代表；不重排原报告原型。</p><p class="en">Two random original regions per class, independent seed20260921, plus the first valid erosion1 assignment switch in original row order. Random selection keeps zero-feature cases. Diagnostic cases are not representative; original prototypes are unchanged.</p>'
    for entry in selections:
        src=root/entry['image_path']
        if hashlib.sha256(src.read_bytes()).hexdigest()!=entry['image_sha256']:raise ValueError('Display source hash')
        im=Image.open(src);w,h=im.size
        if min(w,h)>300:im=im.resize((300,int(300*h/w)) if h>w else (int(300*w/h),300),Image.Resampling.LANCZOS)
        rgb=np.array(im.resize((224,224),Image.Resampling.BILINEAR));p=next((out/'collections'/entry['job']).glob('*'));i=entry['row']
        with np.load(p/'identity0_masks.npz') as z:base=np.unpackbits(z['packed'],axis=2,count=224)[i].astype(bool)
        with np.load(p/'identity0_science.npz') as z:old=int(z['assignments'][i])
        image_id=entry['class_name']+'-'+src.name;target=bundle/'images'/image_id
        if not target.exists():shutil.copy2(src,target)
        body+='<h3>'+html.escape(entry['region_id'])+'</h3><p>'+html.escape(entry['selection_rule'])+' · <a href="images/'+html.escape(image_id)+'">原尺寸源图 / Native source</a></p>'
        for condition in ['erosion1','erosion2','dilation1','dilation2']:
            file=p/(condition+'_science.npz')
            if not file.exists():body+='<p>'+condition+'：未完成 / Not completed.</p>';continue
            with np.load(p/(condition+'_masks.npz')) as z:new=np.unpackbits(z['packed'],axis=2,count=224)[i].astype(bool)
            with np.load(file) as z:assignment=int(z['assignments'][i]);valid=bool(z['valid'][i])
            def overlay(mask):
                a=rgb.copy();a[mask]=(.6*rgb[mask]+.4*np.array([0,195,175])).astype(np.uint8);return a
            selected=rgb.copy();selected[~new]=220;change=np.full_like(rgb,245);change[base&~new]=[205,55,40];change[new&~base]=[45,100,215]
            arrays=[rgb,overlay(base),overlay(new),selected,change];labels=['Original CNN geometry','Baseline: teal selected',condition+': teal selected','New: grey hidden','Red lost / blue added']
            canvas=Image.new('RGB',(5*232,258),'white');draw=ImageDraw.Draw(canvas)
            for col,(a,label) in enumerate(zip(arrays,labels)):canvas.paste(Image.fromarray(a),(col*232,28));draw.text((col*232+3,6),label,fill='#142c35')
            filename=entry['region_id']+'-'+condition+'.png';canvas.save(bundle/'examples'/filename)
            body+='<figure><a href="examples/'+filename+'"><img loading="lazy" width="1160" height="258" src="examples/'+filename+'" alt="Baseline/new selected masks and changed pixels"></a><figcaption>'+condition+' · basis index '+str(old)+' → '+str(assignment)+' (-1 = invalid / 无效) · '+('有效 / valid' if valid else '无效 / invalid')+'</figcaption></figure>'
            rows.append(dict(**entry,condition=condition,old_assignment=old,new_assignment=assignment,valid=valid,display_mask_sha256=hashlib.sha256(new.tobytes()).hexdigest()))
    (bundle/'robustness-display-selection.json').write_text(json.dumps(rows,indent=2))
    return body
