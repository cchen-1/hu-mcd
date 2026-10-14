"""Unify the two accepted medical branches under the existing private report shell."""
from pathlib import Path
import re,json,html,shutil,hashlib
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2];B=R/'artifacts/bunya/ham-derm-discovery-20260924'
def bi(z,e):return '<span data-l="zh">'+z+'</span><span data-l="en" hidden>'+e+'</span>'
def p(z,e,cls=''):return '<p class="'+cls+'">'+bi(z,e)+'</p>'
def h(z,e,n=2):return f'<h{n}>'+bi(z,e)+f'</h{n}>'
def link(url,z,e):return '<a href="'+url+'">'+bi(z,e)+'</a>'
def tabs(pre='',current=''):
 return '<div class="tools medical-subtabs" aria-label="Medical experiment routes">'+''.join('<a class="button" '+('aria-current="page" ' if key==current else '')+'href="'+pre+url+'">'+bi(z,e)+'</a>' for key,url,z,e in [('derm','medical-dermamnist.html','DermaMNIST-C：固定概念外部迁移','DermaMNIST-C: frozen-concept transfer'),('ham','medical-ham-derm/index.html','HAM：Derm7pt 概念拟合','HAM: concept fitting on Derm7pt')])+'</div>'
def table(rows):return '<div class="table-scroll"><table><thead><tr><th>'+bi('比较项','Measure')+'</th><th>DermaMNIST-C</th><th>HAM</th></tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def strip_note(s):return re.sub(r'<!-- HAM-DERM-FIT-CURRENT -->.*?<!-- /HAM-DERM-FIT-CURRENT -->','',s,flags=re.S)
def main_body(s):return s.split('<main',1)[1].split('>',1)[1].split('</main>',1)[0]
def frame(template,body,title,nested=False):
 prefix=template[:template.index('<main')];suffix=template[template.index('</main>')+7:]
 prefix=re.sub(r'<title>.*?</title>','<title>'+title+' | HU-MCD</title>',prefix,flags=re.S)
 # Preserve the shared header/navigation/language control and existing review namespace.
 if nested:
  prefix=re.sub(r'((?:href|src)=")([^"#][^"]*)"',lambda m:m.group(1)+('../'+m.group(2) if not re.match(r'(?:https?:|/)',m.group(2)) else m.group(2))+'"',prefix)
  suffix=re.sub(r'(href=")([^"#][^"]*)"',lambda m:m.group(1)+'../'+m.group(2)+'"',suffix)
 style='<style>.medical-subtabs a[aria-current]{background:#e2eef0;border-color:#087f8c;font-weight:700}.medical-subtabs a{text-decoration:none}.medical-detail .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.medical-detail .grid figure{margin:0}.medical-detail figcaption{overflow-wrap:anywhere;font-size:12px}.medical-detail section{margin-top:32px}.medical-detail{overflow-wrap:anywhere}.medical-detail img{max-width:100%;height:auto}@media(max-width:850px){.medical-detail .grid{grid-template-columns:1fr}}</style>'
 prefix=prefix.replace('</head>',style+'</head>')
 prefix=re.sub(r'(<a )(href="(?:../)?medical.html")', r'\1aria-current="true" \2',prefix)
 return prefix+'<main id="main" class="medical-detail">'+body+'</main>'+suffix

def localize_ham(body):
 body=re.sub(r'<h1>.*?</h1>','',body,count=1,flags=re.S)
 body=re.sub(r'<p><a href="../index.html">.*?</p>','',body,count=1,flags=re.S)
 # Existing paragraphs encode the actual reviewed Chinese and English in that order.
 def paragraphs(m):
  attrs,t=m.group(1),m.group(2)
  if '<br>' in t and re.search('[\u3400-\u9fff]',t):
   z,e=t.split('<br>',1);return '<p'+attrs+'>'+bi(z,e)+'</p>'
  return m.group(0)
 body=re.sub(r'<p([^>]*)>(.*?)</p>',paragraphs,body,flags=re.S)
 # Plain bilingual headings/captions retain meaning and IDs; language switch only hides prose.
 def simple(m):
  tag,attrs,t=m.group(1),m.group(2),m.group(3)
  if '<' not in t and re.search('[\u3400-\u9fff]',t) and re.search(r'\s/\s?',t):
   left,right=re.split(r'\s/\s?',t,maxsplit=1)
   if re.search('[\u3400-\u9fff]',left):z,e=left,right
   else:z,e=right,left
   return '<'+tag+attrs+'>'+bi(z,e)+'</'+tag+'>'
  return m.group(0)
 body=re.sub(r'<(h[2-4]|th|td|summary|li|p)([^>]*)>(.*?)</\1>',simple,body,flags=re.S)
 body=body.replace('COMPLEMENT（非学习概念）',bi('补空间（非学习概念）','Complement (not a learned concept)'))
 body=re.sub(r'<figcaption>(.*?)<br>原选区 / raw pixels: (\d+); 有效选区 / effective224 pixels: (\d+); zero=(True|False)</figcaption>',lambda m:'<figcaption>'+m[1]+'<br>'+bi(f'原选区像素：{m[2]}；有效224选区像素：{m[3]}；零特征：{m[4]}',f'Raw region pixels: {m[2]}; effective224 pixels: {m[3]}; zero feature: {m[4]}')+'</figcaption>',body)
 # This paragraph contains nested links; translate deliberately rather than splitting markup.
 body=re.sub(r'<p>审阅模板：.*?</p>',p('审阅模板：'+link('data/review-template.csv','CSV','CSV')+'；示例清单：'+link('data/example-selection.json','JSON','JSON')+'；'+link('data/discovery-acceptance.json','验收记录','Acceptance')+'；'+link('data/resolved_config.json','配置','Configuration')+'；'+link('data/concept_index.json','完整概念映射','Concept mapping'),'Review template: <a href="data/review-template.csv">CSV</a> · <a href="data/example-selection.json">Example selection</a> · <a href="data/discovery-acceptance.json">Acceptance</a> · <a href="data/resolved_config.json">Configuration</a> · <a href="data/concept_index.json">Concept mapping</a>'),body,flags=re.S)
 body=body.replace('class="notice"','class="note"')
 body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
 return body

def apply(S):
 S=Path(S);template=strip_note((S/'medical.html').read_text())
 original=S/'medical-dermamnist.html'
 if not original.exists():
  original_body=main_body(template)
  original_body=re.sub(r'<div class="eyebrow">.*?</div>','',original_body,count=1,flags=re.S)
  original_body=re.sub(r'<h1>.*?</h1>','',original_body,count=1,flags=re.S)
  original.write_text(frame(template,h('DermaMNIST-C：源域发现与外部迁移','DermaMNIST-C: source discovery and external transfer',1)+p('返回 '+link('medical.html','两条路线比较','route comparison'),'Back to <a href="medical.html">route comparison</a>')+tabs(current='derm')+original_body,'DermaMNIST-C · Medical transfer'))
 ham_path=S/'medical-ham-derm/index.html';ham=main_body(ham_path.read_text())
 if 'data-l="zh"' not in ham:ham=localize_ham(ham)
 else:
  ham=re.sub(r'^.*?</div>(?=\s*<p class="note")','',ham,count=1,flags=re.S)
 ham_path.write_text(frame(template,h('HAM：在 Derm7pt 上拟合概念','HAM: concept fitting on Derm7pt',1)+p('<a href="../medical.html">← 两条路线比较</a>','<a href="../medical.html">← Compare the two routes</a>')+tabs('../','ham')+ham,'HAM · Medical transfer',True))
 # All quantitative claims are frozen accepted evidence, not historical plans.
 new=json.loads((B/'discovery-acceptance.json').read_text());old=json.loads((R/'artifacts/bunya/medical-external-results-20260923/acceptance.json').read_text())['result']
 assert (new['images'],len(new['concepts']),new['zero_features'],new['mel_predicted'])==(101,3,72,34)
 assert (old['images'],old['learned_concepts'],old['zero_features'],old['MEL_predictions'])==(101,1,37,29)
 body='<div class="eyebrow">'+bi('医学迁移 · 已验收结果 · 2026-09-24','Medical transfer · Accepted results · 24 Sep 2026')+'</div>'+h('同一医学方向，两种不同的验证任务','One medical direction, two different tests',1)
 body+=p('两条路线均已完成工程验收。DermaMNIST-C 路线测试“源域概念能否固定后应用到 Derm7pt”；HAM 路线测试“医学分类器配合发布版输入流程，能否在 Derm7pt 上发现概念”。后者不是前者的直接升级或受控对照。','Both routes passed engineering acceptance. DermaMNIST-C tests applying fixed source-domain concepts to Derm7pt; HAM tests concept discovery on Derm7pt using a medical classifier and the released input pipeline. The latter is not a controlled improvement over the former.','lead')
 body+=h('结果比较','Results at a glance')+table([
 [bi('分类器训练来源','Classifier training source'),bi('DermaMNIST-C 224缓存；训练/验证/测试 8,215/573/1,227','DermaMNIST-C 224 cache; train/validation/test 8,215/573/1,227'),bi('原始 HAM 图，经短边300加载后生成分类器224输入；保留同一 corrected split 身份','Original HAM images; short-side300 loading then classifier224 input; same corrected-split identities')],
 [bi('源域测试：准确率 / 平衡准确率','Source test: accuracy / balanced accuracy'),'89.00% / 59.70%','89.32% / 68.97%'],
 [bi('源域测试 MEL 召回','Source-test MEL recall'),'50/70 (71.43%)','45/70 (64.29%)'],
 [bi('验证集选中的 checkpoint','Validation-selected checkpoint'),'Epoch 10','Epoch 27'],
 [bi('概念在哪里拟合','Concept-fitting cohort'),bi('源域400个不同病灶各一图；70图留出','400 source lesions, one image each; 70 held-out images'),bi('Derm7pt R101全部101例；0例留出','All101 Derm7pt R101 cases; no held-out cases')],
 [bi('SAM 前几何','Geometry before SAM'),bi('源域224图；外部采用 E224 方图','Source224 images; external E224 square images'),bi('短边300、保持比例；不额外统一为224方图','Short side300, aspect preserved; no extra224 square resize')],
 [bi('初始聚类 → 学习概念','Initial clusters → learned concepts'),'3 → 1','5 → 3'],
 [bi('学习子空间维数；completeness','Learned dimensions; completeness'),'32; 0.6629','8 / 33 / 21; 0.7853'],
 [bi('Derm7pt 的角色','Role of Derm7pt'),bi('应用已冻结概念，无重拟合','Apply frozen concepts without refitting'),bi('参与拟合与示例展示','Concept fitting and in-sample examples')],
 [bi('Derm7pt 区域：保留 / 零特征','Derm7pt regions: nonzero / zero'),'259 / 37 (296 total; 12.50% zero)','351 / 72 (423 total; 17.02% zero)'],
 [bi('Derm7pt 分配：学习概念 / 补空间','Derm7pt assignment: learned / complement'),'94 / 165','213 / 138'],
 [bi('101例 MEL 中预测为 MEL','Predicted MEL among101 MEL cases'),'29/101 (28.71%)','34/101 (33.66%)'],
 [bi('已完成的征象关联','Completed sign associations'),bi('12项概念＋12项补空间描述性对照','12 concept +12 complement descriptive contrasts'),bi('未运行；不以零填充','Not run; not represented as zero')]
 ])
 body+=p('两种分类器使用相同源图片身份与划分；原图版本不是新的独立受试者队列。但缩放路径、选中 checkpoint、概念拟合集及 SAM 输入几何都不同，因此不能把差异归因于“换成 HAM”一个因素。','The classifiers share source-image identities and split membership; the original-image version is not a new independent participant cohort. Resizing, selected checkpoint, concept-fitting cohort and SAM geometry differ, so changes cannot be attributed solely to “switching to HAM”.','note')
 body+=h('如何解释这些结果','Interpretation')
 body+=p('分类器表现并非全面改善：HAM 的平衡准确率较高，但 MEL 召回从50/70降为45/70。Derm7pt 的34/101也仍低于一半；这首先是固定分类器的行为，不能全部归因于分割或概念发现。','Classifier performance does not improve uniformly: HAM has higher balanced accuracy but lower MEL recall,45/70 versus50/70. Its34/101 MEL predictions on Derm7pt remain below half. This is first a property of the fixed classifier, not something attributable entirely to segmentation or concept discovery.')
 body+=p('HAM 发现3个概念，不等于迁移更成功。拟合集由源域400图改成了Derm7pt101图，区域数和几何也改变；completeness与学习概念分配数较高，只描述当前子空间对当前模型和数据的覆盖，不等于更可理解或医学上更正确。','Finding three concepts with HAM does not establish better transfer. Fitting moved from400 source images to101 Derm7pt images, with changed region counts and geometry. Higher completeness or more learned-concept assignments describes coverage of this model and data, not better understandability or medical validity.')
 body+=p('视觉证据仍需人工审阅。HAM 的有限8例检查中，C002两例包含较大色素区域；C001原型为外围皮肤，而一个随机成员接近图框。C003有78/118初始成员最终归补空间。原型、随机成员和初始聚类必须分开查看，不能直接赋予医学征象名称。','Visual evidence still needs human review. In the limited eight-example HAM inspection, two C002 examples cover large pigmented areas; C001 includes peripheral skin in a prototype but an almost pure frame in a random member.78/118 initial C003 members end in the complement. Inspect prototypes, random members and initial clusters separately before assigning medical sign names.')
 body+=h('疑点与下一步关注','Concerns and next priorities')
 body+=p('① 先审阅概念与选区：区分色素区域、周边皮肤、图框及伪影；背景并不自动等于失败。② 核对零特征和成员迁移：HAM有72个非空掩码产生零特征，排除规则保留，不能靠删例改善结果。③ 保留身份限制：214个相似候选对涉及20个R101病例尚待审阅；病例ID不是患者ID。','1. Review concepts and selected regions: distinguish pigmented areas, surrounding skin, frames and artifacts; background is not automatically failure. 2. Inspect zero features and membership transitions:72 nonempty HAM masks yield zero features; keep the exclusion rule and do not remove examples to improve results. 3. Preserve identity limits:214 similarity candidates involving20 R101 cases await review; case IDs are not patient IDs.')
 body+=p('DermaMNIST-C 的征象结果只是图片级描述性关联，不证明语义等同或空间定位；部分对照只有1个阳性样本。HAM尚无独立留出概念评价。若后续要比较迁移能力，应先约定固定概念、未参与拟合的评价范围和一致处理规则；本次报告更新不启动新实验。','DermaMNIST-C sign results are descriptive image-level associations, not semantic equivalence or spatial localization; some contrasts have only one positive. HAM has no independent held-out concept evaluation. A later transfer comparison needs frozen concepts, evaluation cases unused for fitting and matched processing rules. This report update launches no experiment.')
 body+=h('查看两条实验路线','Explore each experiment')+tabs()+p('子页保留各自完整结果、概念示例、诊断、版本与审阅入口。两条路线的同编号概念没有默认语义对应。','Each subpage retains its results,concept examples,diagnostics,versions and review links. Matching concept numbers across routes do not imply matching semantics.','quiet')
 body+=p('证据：分类器28781915 / 28885816；发现28792059 / 28889071；旧路线外部推理28796209。数值均来自已验收记录，未新增预测。','Evidence: classifiers28781915 /28885816; discovery28792059 /28889071; earlier-route external inference28796209. Values come from accepted records; no new predictions.','quiet')
 (S/'medical.html').write_text(frame(template,body,'Medical transfer · Route comparison'))
 # Preserve the main report wording but point the previous new-branch notice into its parent tab.
 f=S/'index.html';s=f.read_text();s=s.replace('href="medical-ham-derm/index.html"','href="medical.html"');f.write_text(s)
 return {'medical':'medical.html','dermamnist':'medical-dermamnist.html','ham':'medical-ham-derm/index.html'}
if __name__=='__main__':
 import sys
 print(json.dumps(apply(Path(sys.argv[1]))))
