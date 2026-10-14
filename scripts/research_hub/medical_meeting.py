"""English meeting report from accepted results and frozen visual selections."""
from pathlib import Path
import csv,json,hashlib,shutil,re,html
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from .medical_compact import fit,txt,overlay,selected
from .medical_comparison import frame
R=Path(__file__).resolve().parents[2];B=R/'artifacts/bunya'
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rr):return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rr)+'</tbody></table></div>'
def apply(S):
 S=Path(S);out=S/'hub/medical-meeting';out.mkdir(parents=True,exist_ok=True);compact=S/'hub/medical-compact';sel=list(csv.DictReader((compact/'selection.csv').open()));evidence=[]
 H=B/'ham-derm-discovery-20260924';ha=j(H/'discovery-acceptance.json');old=j(B/'medical-external-results-20260923/acceptance.json');assert (ha['images'],ha['held_out'],len(ha['concepts']))==(101,0,3);assert old['result']['learned_concepts']==1
 order=[('derm','HUMCD-MEL-C001','external')]+[('ham',c['concept_id'],'training') for c in sorted(ha['concepts'],key=lambda x:x['global_rank'])]
 # Same frozen members as the reproduction-style compact report, without new sampling.
 image=Image.new('RGB',(2600,2540),'white');txt(image,(35,20),'Medical HU-MCD | All four learned concepts across two experiments',39);txt(image,(35,79),'P1/P2: saved prototypes   R1/R2: saved random members   Cyan SELECTED; grey HIDDEN',27)
 for i,(route,cid,role) in enumerate(order):
  y=140+i*580;txt(image,(35,y),cid,31);txt(image,(35,y+41),'DermaMNIST-C | EXTERNAL members; concepts fitted on source data' if route=='derm' else 'HAM classifier | Derm7pt FIT members; no held-out concept validation',25)
  chosen=[]
  for view in ['top','random']:
   selected_rows=[x for x in sel if x['route']==route and x['role']==role and x['view']==view and x.get('display_concept')==cid and x['selection']=='first 2 in frozen saved selection'];assert len(selected_rows)==2;chosen+=selected_rows
  for k,r in enumerate(chosen):
   x=35+k*645;asset=compact/r['asset'];im=Image.open(asset).convert('RGB');txt(image,(x,y+82),('P' if k<2 else 'R')+str(k%2+1),24);fit(image,im,(x,y+115,610,370));txt(image,(x,y+494),r['image_id']+' / S'+str(r['raw_index']).zfill(4)+' | '+f"{float(r['effective_fraction']):.1%} selected",22)
   evidence.append(r|dict(figure='concept-summary.png',asset_sha256=sha(asset)))
 txt(image,(35,2472),'Different fitting cohorts, classifiers and geometries; rows are not a controlled ranking of methods.',26)
 txt(image,(35,2510),'Complements are separate in the detailed reports. No clinical names assigned. Display enlargement adds no image detail.',24)
 image.save(out/'concept-summary.png',dpi=(300,300))
 # The already-fixed case578; preserve original indices and identify each mask stage.
 image=Image.new('RGB',(2600,1390),'white');txt(image,(35,20),'Same case, different stages | Derm7pt-case578',41);txt(image,(35,80),'Raw SAM proposals and final assignments are different outputs. This case is illustrative, not a quality benchmark.',26)
 for i,route in enumerate(['derm','ham']):
  base=B/'medical-external-results-20260923/collection' if route=='derm' else H/'collection-28889071';cache=base/'cache' if route=='derm' else base/'cache/training';iid='Derm7pt-case578'
  inp=(base/'inputs'/f'{iid}.png') if route=='derm' else H/'collection-28883445/derm7pt_r101_presam'/f'{iid}.png';im=Image.open(inp).convert('RGB');raw=np.load(cache/(iid+'_sam_raw.npz'))['masks'];em=np.load(cache/(iid+'_effective_masks.npz'))['masks'];central=1 if route=='derm' else 2
  y=150+i*555;txt(image,(35,y),'DermaMNIST-C | external application | E224' if route=='derm' else 'HAM | fitting on Derm7pt | short side 300, aspect preserved',31)
  tiles=[(im,'Actual SAM input',f'{im.width} x {im.height}'),(overlay(im,raw[0]),'Raw SAM candidate 000','Example candidate; not a clinical annotation'),(selected(im,em[0]),'Final peripheral region S0000','COMPLEMENT' if route=='derm' else 'C002'),(selected(im,em[central]),f'Final central region S{central:04d}','C001' if route=='derm' else 'C002')]
  for k,(v,title,sub) in enumerate(tiles):
   x=35+k*645;txt(image,(x,y+62),title,26);fit(image,v,(x,y+106,605,350));txt(image,(x,y+477),sub,23)
  for k in [0,central]:
   rec=next(x for x in sel if x['route']==route and x['view']=='pipeline_all_regions' and x['image_id']==iid and int(x['raw_index'])==k)
   assert hashlib.sha256(em[k].tobytes()).hexdigest()==rec['effective_mask_sha256'];evidence.append(rec|dict(figure='same-case.png',raw_sam_candidate_index=0,raw_sam_mask_sha256=hashlib.sha256(raw[0].tobytes()).hexdigest()))
 txt(image,(35,1290),'The displayed raw candidate is not a one-to-one parent of both final regions. HU-MCD postprocessing recomposes masks.',24)
 txt(image,(35,1330),'Grey = hidden for display only. Effective 224 masks are mapped back for display; no new inference or input modification.',24)
 image.save(out/'same-case.png',dpi=(300,300))
 figs=[dict(file=n,width=Image.open(out/n).width,height=Image.open(out/n).height,sha256=sha(out/n)) for n in ['concept-summary.png','same-case.png']]
 (out/'evidence.json').write_text(json.dumps(dict(selection='Existing frozen P1/P2/R1/R2; previously fixed first cohort case578. SAM000 is an illustrative previously shown candidate, not an accuracy-selected sample.',records=evidence,figures=figs,new_predictions=0,new_sampling=0),indent=2)+'\n')
 def fig(name,caption):return '<figure id="'+name.split('.')[0]+'"><a href="hub/medical-meeting/'+name+'"><img src="hub/medical-meeting/'+name+'" alt="'+html.escape(caption)+'"></a><figcaption>'+caption+' Click for full-resolution export.</figcaption></figure>'
 body='<div class="eyebrow">MEETING REPORT · ACCEPTED RESULTS THROUGH 24 SEPTEMBER 2026 · PRIVATE</div><h1>HU-MCD in dermatology: what worked, and what remains unproven</h1>'
 body+='<p class="lead">We completed two melanoma-focused pilots. Both produced traceable regions and concepts. The current evidence supports technical feasibility, but does not yet establish accurate medical segmentation or clinically meaningful concept identities.</p><p><a href="medical.html">Medical transfer overview</a> · <a href="hub/medical-meeting/meeting-report.md">English Markdown</a> · <a href="#setup">1. Design</a> · <a href="#results">2. Results</a> · <a href="#discussion">3. Discussion</a></p>'
 body+='<section id="setup"><h2>1. What we did and why</h2><p><strong>Research question:</strong> can the released HU-MCD pipeline explain a fixed medical classifier using concepts that relate to dermatological content? We started with the approved melanoma (MEL) group rather than all diagnoses. The classifier still predicts seven skin-lesion classes; concept discovery targets its MEL output.</p>'
 body+='<p><strong>Why DermaMNIST-C first?</strong> It provided a traceable 224-pixel release and a documented corrected split for a manageable first pilot. We trained a standard, ImageNet-initialized ResNet50 so that architecture, training data and weights were controlled and recorded. This was a practical starting point, not evidence that 224-pixel images were optimal for SAM.</p>'
 body+='<p><strong>Why then original HAM10000?</strong> We wanted to use original images and follow the released HU-MCD loading path more closely: short side capped at300, aspect ratio preserved, with no extra square224 resize before SAM. We also changed the discovery task to fitting concepts directly on Derm7pt. This was a second feasibility experiment, not a controlled test of one change.</p>'
 body+='<p><strong>The datasets are related.</strong> In our accepted releases, DermaMNIST-C and original HAM share all10,015 source-image identities and labels. We retained the same corrected train/validation/test membership (8,215/573/1,227). Direct bicubic resizing of the originals reproduced all10,015 accepted DermaMNIST-C224 images. The new loading path changes pixels through a different resize sequence; it does not introduce a new independent cohort. <a href="medical-ham-derm/data/classifier-acceptance.json">Classifier receipt</a>; <a href="hub/medical-meeting/data-acceptance.json">source-image audit</a>.</p>'
 body+=table(['Setting','A · DermaMNIST-C','B · Original HAM'],[
 ['Classifier','ImageNet-initialized ResNet50; selected epoch10','ImageNet-initialized ResNet50; selected epoch27'],
 ['Concept fitting','400 source MEL images, one per lesion; fixed seed43','All101 fixed Derm7pt R101 MEL cases'],
 ['Images outside concept fitting','70 source test images (61 lesions); 101 Derm7pt cases without refitting','None: all101 Derm7pt cases participated in fitting'],
 ['Pre-SAM geometry','224×224 source images; approved E224 external warp','Short side300; aspect preserved'],
 ['Purpose','Apply source-fitted concepts to another dataset','Test concept discovery on Derm7pt with the released loading path']])
 body+='<p class="quiet">Shared training recipe: 50 epochs, batch64, seed43, ordinary cross-entropy, Adam1e−4; checkpoint selected using validation macro one-vs-rest AUC. Classifiers were then frozen. One seed explains one fixed classifier, not training-seed stability. HU-MCD retained SAM ViT-H and the released clustering, subspace and assignment rules, including minimum cluster size50. Classifier resizing to224 remains a later step and is distinct from pre-SAM geometry.</p></section>'
 body+='<section id="results"><h2>2. Results: one source-fitted concept and three Derm7pt-fitted concepts</h2>'
 body+=table(['Discovery result','A · DermaMNIST-C','B · HAM → Derm7pt'],[
 ['Fitting images','400','101'],['Initial clusters → retained learned concepts','3 → 1','5 → 3'],['Postprocessed regions → nonzero features','917 → 849','423 → 351'],['Nonempty-mask zero features','68/917 (7.42%)','72/423 (17.02%)'],['Completeness','0.6629','0.7853']])
 body+='<p>Small clusters were filtered by the existing rule; concepts were not selected for appearance. Completeness measures classifier-weight subspace coverage, not medical meaning or human interpretability. Both runs passed engineering acceptance; this is not medical validation.</p>'
 body+=table(['Concept ID','Initial cluster / members','Dimension','Final assigned regions'],[
 ['HUMCD-MEL-C001','0 / 807','32','514 source fit; 82 source held-out; 94 external'],
 *[[c['concept_id'],str(c['cluster_id'])+' / '+str(c['initial_members']),c['subspace_dimension'],str(c['final_members'])+' Derm7pt fit'] for c in sorted(ha['concepts'],key=lambda x:x['global_rank'])]])
 body+='<p>The complement is not a learned concept. Its assignments are335 source-fit /61 held-out /165 external in A, and138 fitting regions in B. Final assignment counts differ from initial cluster sizes. Stable IDs do not imply matching meanings between experiments.</p>'
 body+=fig('concept-summary.png','Figure 1. Every learned concept: two saved prototypes (P) and two saved random members (R). A uses external members; B uses fitting members. No visual-quality selection or semantic matching. Cyan marks the effective selected region; grey marks hidden context.')
 body+='<p><a href="medical-dermamnist.html#medical-concepts">A: source, held-out and external members together</a> · <a href="medical-ham-derm/index.html#medical-concepts">B: all concepts and initial-member inspection</a> · <a href="medical-dermamnist.html#medical-oc">A complement</a> · <a href="medical-ham-derm/index.html#medical-oc">B complement</a>. These views preserve all saved examples and exploratory review forms.</p>'
 body+='<h3>The classifiers remain a separate limitation</h3>'+table(['Fixed-classifier result','A','B'],[['Source-test MEL predictions','50/70 (71.43%)','45/70 (64.29%)'],['Derm7pt R101 MEL predictions','29/101 (28.71%)','34/101 (33.66%)']])
 body+='<p>These are unmasked full-image predictions. SAM and concept assignment did not cause these classification outcomes. R101 contains only the approved MEL group, so these fractions do not measure seven-class external accuracy or specificity. The source and Derm7pt cohorts differ.</p></section>'
 body+='<section id="discussion"><h2>3. What the evidence supports</h2>'
 body+='<h3>SAM proposes distinct regions; final concepts can still group different content</h3>'
 body+=fig('same-case.png','Figure 2. The same fixed case578 under both routes. A assigns the central region to C001 and the peripheral region to the complement. B assigns both regions to C002. Raw SAM candidate000 is shown as an example, not as medical ground truth. This is an illustrative case, not a representative quality estimate.')
 body+='<p><strong>Supported:</strong> the saved masks and assignment arrays show that distinct regions can share a final concept. In B, case578-S0000 (mainly peripheral) and S0002 (central darker region) both map toC002. <strong>Not established:</strong> why this happens across the dataset, whether either region has an expert-defined medical identity, or whether SAM has high boundary accuracy. SAM output, HU-MCD mask postprocessing and feature-space assignment must be examined separately.</p>'
 body+='<h3>Prototypes alone give an incomplete picture</h3><p>For A’s external C001, the displayed prototypes case760-S0000 and case793-S0002 cover81.3% and88.5% of the image. The displayed random members case578-S0001 and case781-S0001 cover19.5% and23.8%. They show different spatial extents; area differences alone do not prove semantic inconsistency. This is why we retain random members rather than naming a concept from its best-looking prototypes.</p>'
 body+='<p>For B,78/118 initial members of the cluster supporting C003 finally map to the complement. This documents a substantial membership change, not a measured assignment error. <a href="medical-ham-derm/data/cluster-to-final.csv">Cluster-to-assignment table</a>.</p>'
 body+='<h3>Clinical association is preliminary</h3><p>A completed12 prespecified image-level sign contrasts, plus12 separate complement contrasts. For irregular dots/globules, C001 mean coverage was19.77 percentage points higher than in the absent group (84 versus12 cases; descriptive AUC0.660). This is one of12 results, not a confirmed discovery. Two other contrasts have only one positive case. B’s sign analysis has not been run.</p><p>Derm7pt’s seven-point labels support image-level comparisons here, not pixel-level localization. We have not shown that a concept <em>is</em> a named sign or locates that sign. No clinician concept review or expert-mask segmentation benchmark has been completed. <a href="medical.html">All12 contrasts and interpretation limits</a>; <a href="hub/medical-meeting/contrasts.csv">full-precision results, including complement</a>.</p>'
 body+='<h3>Neither the route comparison nor the ImageNet comparison establishes better concept quality</h3><p>B found three concepts and A found one, but classifier weights, loading geometry and fitting cohort all changed. A tests application of frozen source concepts; B uses Derm7pt for fitting. The ImageNet reproduction found147 learned concepts across10 classes, but no common human-quality evaluation compares them with these medical concepts. More concepts or higher completeness is not evidence of better medical interpretation.</p>'
 body+='<p class="note"><strong>Meeting conclusion:</strong> HU-MCD can be executed on these medical classifiers with traceable image-to-region-to-concept outputs. The main open question is whether the resulting groups have stable, clinically meaningful content. The current images motivate that question; they do not settle it.</p>'
 body+='<p><strong>Next evidence needed:</strong> review initial and final random members to locate where content mixing appears; obtain expert review or pixel annotations for a clearly defined segmentation target; and use a separately frozen held-out protocol before claiming concept generalization for B. These are proposed next steps, not experiments executed for this report.</p>'
 body+='<details><summary>Traceability and limits retained</summary><p>Accepted data and model provenance are available. Exact-match checks did not identify a source–Derm7pt duplicate under the audited definitions;214 similarity candidate pairs involving20 R101 cases remain unconfirmed. Case IDs are not patient IDs. All cases, zero-feature records, warnings and failure history remain. Neither numerical reconstruction nor full image coverage proves medical semantics.</p><p>Scientific jobs: A classifier28781915, discovery28792059, external28796209; B classifier28885816, discovery28889071. No new model execution, fitting or display sampling. Figures reuse fixed selections from the existing report.</p><p><a href="hub/medical-meeting/evidence.json">Figure IDs, masks and hashes</a> · <a href="hub/medical-meeting/provenance.json">Source receipts, execution versions and generation record</a></p></details></section>'
 # Readable English spacing; stable IDs and source paths remain literal.
 for before,after in [('all10,015','all 10,015'),('DermaMNIST-C224','DermaMNIST-C 224'),('at300','at 300'),('square224','square 224'),('epoch10','epoch 10'),('epoch27','epoch 27'),('All101','All 101'),('all101','all 101'),('side300','side 300'),('batch64','batch 64'),('seed43','seed 43'),('Adam1e','Adam 1e'),('size50','size 50'),('to224','to 224'),('are335','are 335'),('/61 held','/ 61 held'),('/165 external','/ 165 external'),('and138','and 138'),('toC002','to C002'),('B,78/118','B, 78/118'),('completed12','completed 12'),('plus12','plus 12'),('was19.77','was 19.77'),('versus12','versus 12'),('AUC0.660','AUC 0.660'),('of12','of 12'),('All12','All 12'),('found147','found 147'),('across10','across 10'),(';214','; 214'),('involving20','involving 20'),('classifier28781915','classifier 28781915'),('discovery28792059','discovery 28792059'),('external28796209','external 28796209'),('classifier28885816','classifier 28885816'),('discovery28889071','discovery 28889071'),('case578','case 578'),('case760','case 760'),('case793','case 793'),('case781','case 781'),('cover81.3','cover 81.3'),('and88.5','and 88.5'),('cover19.5','cover 19.5'),('and23.8','and 23.8'),('candidate000','candidate 000')]:
  body=body.replace(before,after)
 template=(S/'medical.html').read_text();page=frame(template,body,'Medical transfer · Meeting report (English)')
 # Explicit English manuscript; keep the shared navigation but remove a misleading language toggle for this page.
 page=re.sub(r'<button\b[^>]*id="language"[^>]*>.*?</button>','<span class="meeting-language">English meeting report</span>',page,flags=re.S)
 page=re.sub(r'<html([^>]*)lang="[^"]+"',r'<html\1lang="en"',page)
 page=page.replace('</head>','<style>@media print{header,nav,footer,.tools{display:none}main{max-width:none}h2,h3{break-after:avoid}figure{break-inside:avoid}body{font-size:11pt}.meeting-language{color:white}}</style></head>')
 page=page.replace('src="hub/app.js"','src="hub/app.js"')
 # Set URL language before the shared deferred app script reads it; preserve other query parameters/hash.
 page=page.replace('</head>','<script>try{const u=new URL(location.href);u.searchParams.set("lang","en");history.replaceState(null,"",u)}catch(e){}</script></head>')
 (S/'medical-meeting.html').write_text(page)
 # Markdown is an English, editable companion, generated from the same content, not a separate narrative.
 from html.parser import HTMLParser
 class MD(HTMLParser):
  def __init__(self):super().__init__();self.out=[];self.link=[];self.table=False;self.row=[];self.cell=None;self.heading=False
  def handle_starttag(self,t,a):
   a=dict(a)
   if t in ['h1','h2','h3']:self.out.append('\n\n'+'#'*int(t[1])+' ')
   elif t=='p':self.out.append('\n\n')
   elif t=='strong':self.out.append('**')
   elif t=='em':self.out.append('*')
   elif t=='a':self.link.append(a.get('href',''));self.out.append('[')
   elif t=='img':self.out.append('!['+a.get('alt','')+']('+a.get('src','')+')')
   elif t=='table':self.table=True;self.out.append('\n\n')
   elif t=='tr':self.row=[]
   elif t in ['th','td']:self.cell='';self.heading=t=='th'
   elif t=='summary':self.out.append('\n\n**')
  def handle_data(self,d):
   if self.cell is not None:self.cell+=d
   else:self.out.append(d)
  def handle_endtag(self,t):
   if t in ['strong','summary']:self.out.append('**')
   elif t=='em':self.out.append('*')
   elif t=='a':self.out.append(']('+self.link.pop()+')')
   elif t in ['th','td']:self.row.append(self.cell);self.cell=None
   elif t=='tr':
    self.out.append('| '+' | '.join(self.row)+' |\n')
    if self.heading:self.out.append('| '+' | '.join(['---']*len(self.row))+' |\n')
   elif t=='table':self.table=False;self.out.append('\n')
 md=MD();md.feed(body);text=''.join(md.out).replace('](#setup)','](#1-what-we-did-and-why)').replace('](#results)','](#2-results-one-source-fitted-concept-and-three-derm7pt-fitted-concepts)').replace('](#discussion)','](#3-what-the-evidence-supports)')
 # Companion is nested two levels under the report root; preserve correct offline links.
 text=re.sub(r'\]\((?!https?:|#)([^)]+)\)',lambda m:'](../../'+m[1]+')',text)
 (out/'meeting-report.md').write_text(text.strip()+'\n')
 shutil.copy2(H/'data-acceptance.json',out/'data-acceptance.json');shutil.copy2(B/'medical-external-results-20260923/contrasts.csv',out/'contrasts.csv')
 sources=[H/'discovery-acceptance.json',H/'classifier-acceptance.json',H/'data-acceptance.json',B/'medical-external-results-20260923/acceptance.json',B/'batch-review-20260922/acceptance-B.json']
 provenance=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),scientific_evidence_date='2026-09-24',generator=str(Path(__file__).relative_to(R)),generator_sha256=sha(Path(__file__)),sources=[dict(path=str(p.relative_to(R)),sha256=sha(p)) for p in sources],execution_commits=dict(A_classifier='00f06dc549439db92e4842cb5ecd761f9b38bc43',A_discovery='5623abe8a843deaea7ab838a8c76c41afbad565e',A_external='dcc61934973aa9054a4e689ccf2334652447c575',B_classifier='4a7360c31e41fb88f94734b0cd8167ae497c9b42',B_discovery='649ce176c7d2e4e17b9e9be196fb0e7c33ef3e42'),new_predictions=0,new_jobs=0,new_sampling=0,figures=figs)
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 for name in ['medical.html','medical-dermamnist.html','medical-ham-derm/index.html']:
  f=S/name;s=f.read_text();s=re.sub(r'<!-- MEDICAL-MEETING-LINK -->.*?<!-- /MEDICAL-MEETING-LINK -->','',s,flags=re.S);href='../medical-meeting.html?lang=en' if name.startswith('medical-ham') else 'medical-meeting.html?lang=en';link='<p class="quiet"><a href="'+href+'"><span data-l="zh">英文会议汇报：设计、概念汇总与证据讨论</span><span data-l="en" hidden>English meeting report: design, concept summary and evidence</span></a></p>';s=s.replace('</h1>','</h1><!-- MEDICAL-MEETING-LINK -->'+link+'<!-- /MEDICAL-MEETING-LINK -->',1);f.write_text(s)
 return provenance
