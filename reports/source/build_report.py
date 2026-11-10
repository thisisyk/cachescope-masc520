"""Build the anonymous editable IEEE-style course report from saved prose/data.

Requires python-docx, ReportLab and Node.js with sharp. The measured datasets and
model artifacts are read only. Native Word exports the companion PDF separately.
"""
from __future__ import annotations
import json, os, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'reports'; SOURCE=REPORTS/'source'; FIG=SOURCE/'figures'
METRICS=ROOT/'experiments/v3/results/metrics.json'
NODE=os.environ.get('REPORT_NODE','node')
BLACK='000000'; BLUE=HexColor('#24485f'); GRAY=HexColor('#c7cdd1')
for directory in [REPORTS,SOURCE,FIG]: directory.mkdir(parents=True,exist_ok=True)
if Path('C:/Windows/Fonts/arial.ttf').exists(): pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
FONT='Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

def text(d,x,y,s,size=8.5,anchor='start',fill=None):
    d.add(String(x,y,s,fontName=FONT,fontSize=size,textAnchor=anchor,fillColor=fill or HexColor('#111111')))

def export_drawing(d,name):
    svg=FIG/(name+'.svg'); png=FIG/(name+'.png')
    renderSVG.drawToFile(d,str(svg))
    script="const sharp=require('sharp');sharp(process.argv[1],{density:360}).png().toFile(process.argv[2]).catch(e=>{console.error(e);process.exit(1)});"
    subprocess.run([NODE,'-e',script,str(svg),str(png)],check=True)
    return png

def figures(metrics):
    d=Drawing(250,177)
    nodes=[(4,147,242,26,['Original context and question']),
           (4,113,242,26,['Fixed 60% and 30% candidates']),
           (4,73,242,32,['Pre-answer features and learned risk','Optional JSON syntax check']),
           (4,37,242,28,['Shortest eligible candidate','or original-context fallback']),
           (4,3,242,26,['One answering-model call'])]
    for x,y,w,h,lines in nodes:
        d.add(Rect(x,y,w,h,strokeColor=HexColor('#8d969c'),fillColor=HexColor('#ffffff'),strokeWidth=.65))
        for j,line in enumerate(lines): text(d,x+w/2,y+h/2+(len(lines)-1)*4.8-j*9.6-3,line,8.6,'middle')
    for upper,lower in zip(nodes,nodes[1:]):
        y1=upper[1]; y2=lower[1]+lower[3]
        d.add(Line(125,y1,125,y2+2,strokeColor=BLUE,strokeWidth=.75))
        d.add(Polygon([122,y2+4,128,y2+4,125,y2],fillColor=BLUE,strokeColor=BLUE))
    flow=export_drawing(d,'system_flow')
    d=Drawing(250,166)
    names=[('original','Original'),('always_60','Fixed keep 60%'),('always_30','Fixed keep 30%'),('frozen_v2_mlp','Frozen V2 MLP'),('mlp_seed_520','New MLP seed 520'),(metrics['primary_policy'],'Locked hybrid')]
    x0=95; scale=3.35; base=32
    for tick in [0,10,20,30,40]:
        xx=x0+tick*scale;d.add(Line(xx,base-2,xx,154,strokeColor=HexColor('#e4e7e9'),strokeWidth=.4));text(d,xx,19,str(tick),7.8,'middle')
    for i,(key,label) in enumerate(names):
        y=142-i*19;value=metrics['policies'][key]['harmful_accepted_count']
        text(d,x0-5,y+2,label,8.4,'end')
        color=BLUE if key==metrics['primary_policy'] else HexColor('#5f6d76') if key=='mlp_seed_520' else GRAY
        if value: d.add(Rect(x0,y,value*scale,10,fillColor=color,strokeColor=color))
        text(d,x0+value*scale+4,y+1,str(value),8.4)
    d.add(Line(x0,base-2,x0+40*scale,base-2,strokeColor=HexColor('#555555'),strokeWidth=.6))
    text(d,x0+67,7,'Harmful compressions among 200 tasks',8.2,'middle')
    harm=export_drawing(d,'harm_comparison')
    return flow,harm

def style_font(style,size,bold=False,italic=False):
    style.font.name='Times New Roman';style.font.size=Pt(size);style.font.bold=bold;style.font.italic=italic;style.font.color.rgb=RGBColor(0,0,0)
    rf=style.element.get_or_add_rPr().get_or_add_rFonts()
    for name in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+name),'Times New Roman')
    for name in ['asciiTheme','hAnsiTheme','eastAsiaTheme','csTheme']:rf.attrib.pop(qn('w:'+name),None)

def set_cols(section,num):
    cols=section._sectPr.find(qn('w:cols'))
    if cols is None: cols=OxmlElement('w:cols');section._sectPr.append(cols)
    cols.set(qn('w:num'),str(num));cols.set(qn('w:space'),'360');cols.set(qn('w:equalWidth'),'1')

def paragraph(doc,text,style='Normal',align=None,keep=None):
    p=doc.add_paragraph(style=style)
    clean=text.replace('`','').replace('**','').replace('–','-').replace('—','-').replace('−','-')
    p.add_run(clean)
    if align is not None:p.alignment=align
    if keep is not None:p.paragraph_format.keep_with_next=keep
    return p

def image(doc,path,caption):
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing=1.0
    p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(3.46))
    p=paragraph(doc,caption,'Caption',WD_ALIGN_PARAGRAPH.JUSTIFY)
    p.paragraph_format.space_after=Pt(6)

def table(doc,rows,kind):
    if kind==1:
        rows=[['Component','Tasks','Sources','Groups','Use'],['Earlier data','960','910','125','Development'],['New calibration','160','160','32','Select threshold'],['New final test','200','200','40','Evaluate'],['Total','1,320','1,270','-','2,640 candidates']]
        widths=[.90,.43,.50,.43,1.24]
    elif kind==2:
        aliases={'Original':'Original','Fixed keep 60%':'Fixed keep 60%','Fixed keep 30%':'Fixed keep 30%','JSON guard only':'JSON guard only','Frozen V2 MLP':'Frozen V2 MLP','Locked V2 MLP plus guard':'Locked hybrid','New MLP seed 520':'New MLP seed 520','New maximum-score ensemble':'New max ensemble','New LR':'New LR','New RF':'New RF'}
        rows=[[aliases.get(c,c) if j==0 else c for j,c in enumerate(row)] for row in rows]
        rows[0]=['Policy','Acc %','Save %','Harm','Gain']
        widths=[1.55,.49,.52,.46,.48]
    else:
        widths=[2.78,.76,.94,.91,.86,1.00]
    t=doc.add_table(rows=1,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for c,w in zip(t.columns,widths): c.width=Inches(w)
    for i,row in enumerate(rows):
        cells=t.rows[0].cells if i==0 else t.add_row().cells
        for j,(cell,value) in enumerate(zip(cells,row)):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
            for tag,n in [('top',55),('bottom',55),('left',50),('right',50)]:
                item=OxmlElement('w:'+tag);item.set(qn('w:w'),str(n));item.set(qn('w:type'),'dxa');mar.append(item)
            tcPr.append(mar)
            if i==0:
                shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'E9EDF0');tcPr.append(shading)
            p=cell.paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=Pt(10)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or (kind==1 and j==4) else WD_ALIGN_PARAGRAPH.CENTER
            r=p.add_run(value);r.font.name='Times New Roman';r.font.size=Pt(9 if kind==3 else 8.4);r.bold=i==0
        trPr=t.rows[i]._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
        if i==0:trPr.append(OxmlElement('w:tblHeader'))
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
    t._tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)

REFERENCES=[
 '[1] Y. Li, B. Dong, F. Guerin, and C. Lin, "Compressing Context to Enhance Inference Efficiency of Large Language Models," in Proc. EMNLP, pp. 6342-6353, 2023. doi: 10.18653/v1/2023.emnlp-main.391.',
 '[2] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, and L. Qiu, "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models," in Proc. EMNLP, pp. 13358-13376, 2023. doi: 10.18653/v1/2023.emnlp-main.825.',
 '[3] Z. Pan et al., "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression," in Findings ACL, pp. 963-981, 2024. doi: 10.18653/v1/2024.findings-acl.57.',
 '[4] Y. Geifman and R. El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option," in Proc. ICML, PMLR, vol. 97, pp. 2151-2159, 2019. https://proceedings.mlr.press/v97/geifman19a.html.',
 '[5] B. Hui et al., "Qwen2.5-Coder Technical Report," arXiv:2409.12186, 2024. https://arxiv.org/abs/2409.12186.',
 '[6] OpenAI, "Codex," software, accessed Nov. 11, 2026. [Online]. Available: https://openai.com/codex/.'
]

def main():
    metrics=json.loads(METRICS.read_text(encoding='utf-8'))
    flow,harm=figures(metrics)
    doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=Inches(.75);sec.bottom_margin=Inches(1);sec.left_margin=Inches(.625);sec.right_margin=Inches(.625)
    sec.header_distance=Inches(.25);sec.footer_distance=Inches(.45);set_cols(sec,1)
    for name in ['Normal','Body Text']:
        style_font(doc.styles[name],10);p=doc.styles[name].paragraph_format;p.line_spacing=Pt(12);p.space_after=Pt(3);p.first_line_indent=Inches(.14);p.widow_control=True;p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    style_font(doc.styles['Title'],24);doc.styles['Title'].paragraph_format.space_after=Pt(5)
    for name in ['Heading 1','Heading 2']:
        style_font(doc.styles[name],10,italic=name=='Heading 2');p=doc.styles[name].paragraph_format;p.first_line_indent=Pt(0);p.space_before=Pt(7);p.space_after=Pt(3);p.keep_with_next=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER if name=='Heading 1' else WD_ALIGN_PARAGRAPH.LEFT
    style_font(doc.styles['Caption'],8.5);cp=doc.styles['Caption'].paragraph_format;cp.first_line_indent=Pt(0);cp.line_spacing=Pt(10);cp.space_before=Pt(3);cp.space_after=Pt(4)
    style_font(doc.styles['Subtitle'],10);doc.styles['Subtitle'].paragraph_format.space_after=Pt(6)
    paragraph(doc,'CacheScope Neural Risk Prediction\nfor Prompt Compression','Title',WD_ALIGN_PARAGRAPH.CENTER)
    paragraph(doc,'MASC 520 Final Project Report','Subtitle',WD_ALIGN_PARAGRAPH.CENTER)
    sec=doc.add_section(WD_SECTION_START.CONTINUOUS);set_cols(sec,2)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
    lines=(SOURCE/'manuscript.md').read_text(encoding='utf-8').splitlines()
    i=0;table_no=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line.startswith('# ') or line=='MASC 520 Course Project':continue
        if line.startswith('## Reference integration'):break
        if line=='## Abstract':
            p=paragraph(doc,'Abstract','Heading 2');p.runs[0].bold=True
            continue
        if line.startswith('## '):
            paragraph(doc,line[3:].upper(),'Heading 1');continue
        if line.startswith('### '):
            paragraph(doc,line[4:],'Heading 2');continue
        if line.startswith('**Table '):
            caption=line.replace('**','').replace('Table I.','TABLE I').replace('Table II.','TABLE II')
            paragraph(doc,caption,'Caption',WD_ALIGN_PARAGRAPH.CENTER,True);continue
        if line.startswith('|'):
            rows=[]
            while True:
                if not re.match(r'^\|\s*[-:]+',line):rows.append([s.strip() for s in line.strip('|').split('|')])
                if i>=len(lines) or not lines[i].strip().startswith('|'):break
                line=lines[i].strip();i+=1
            table_no+=1;table(doc,rows,table_no)
            continue
        if line.startswith('Index Terms:'):
            p=paragraph(doc,line);p.paragraph_format.first_line_indent=Pt(0)
            for r in p.runs:r.italic=True;r.font.size=Pt(9)
            continue
        p=paragraph(doc,line)
        if line.startswith('For source context x,'):
            p.text='The observed regression label is true when an original prompt produces a correct answer but a compressed candidate produces an incorrect answer. A conditional safe example has correct answers on both prompts. Candidates whose original answer is incorrect are excluded from risk-classifier fitting: failure on the original cannot establish preservation of a correct answer. They remain in every task-level policy evaluation, including cases improved by compression.'
        if line.endswith('Running the original and both candidates is an offline labeling expense.'):
            image(doc,flow,'Fig. 1. CacheScope chooses a prompt before answering. The original prompt is retained when no candidate is eligible.')
        if line.endswith('best test row.'):
            image(doc,harm,'Fig. 2. Harmful compressions on the same 200 tasks. Token budgets and acceptance rates differ between policies.')
    paragraph(doc,'REFERENCES','Heading 1')
    for ref in REFERENCES:
        p=paragraph(doc,ref);p.paragraph_format.first_line_indent=Inches(-.12);p.paragraph_format.left_indent=Inches(.12);p.paragraph_format.line_spacing=Pt(10.5)
        for r in p.runs:r.font.size=Pt(8.5)
    final_section=doc.add_section(WD_SECTION_START.CONTINUOUS);set_cols(final_section,1)
    doc.add_page_break()
    paragraph(doc,'APPENDIX A COMPLETE FINAL TEST RESULTS','Heading 1')
    paragraph(doc,'All sixteen policies below were specified before final evaluation and use the same 200 tasks. The asterisk identifies the calibration-selected primary. Accuracy and saving are percentages; accepted, harmful, and improved are task counts. This appendix retains comparisons that miss a target or outperform the primary on an individual metric.')
    paragraph(doc,'TABLE III Complete frozen-policy outcomes','Caption',WD_ALIGN_PARAGRAPH.CENTER,True)
    labels={'original':'Original','always_60':'Fixed keep 60%','always_30':'Fixed keep 30%','simple_rule_gate':'Simple retention rule','mlp_seed_520':'New MLP seed 520','mlp_seed_521':'New MLP seed 521','mlp_seed_522':'New MLP seed 522','mlp_ensemble':'New maximum-score ensemble','hybrid_mlp_ensemble':'New ensemble plus JSON check','lr':'New logistic regression','rf':'New random forest','hybrid_lr':'New LR plus JSON check','hybrid_rf':'New RF plus JSON check','hybrid_frozen_v2_mlp':'Frozen V2 plus JSON check *','guard_only':'JSON check only','frozen_v2_mlp':'Frozen V2 original threshold'}
    all_rows=[['Policy','Acc %','Save %','Accepted','Harmful','Improved']]
    for key,m in metrics['policies'].items():all_rows.append([labels.get(key,key),f"{100*m['accuracy']:.2f}",f"{100*m['input_token_reduction_ratio']:.2f}",str(m['accepted_count']),str(m['harmful_accepted_count']),str(m['improvement_count'])])
    table(doc,all_rows,3)
    paragraph(doc,'REPRODUCING THE SAVED EXPERIMENT','Heading 1')
    paragraph(doc,'The public repository contains all three experimental versions. The final data, weights, frozen selection, case studies, and verification files are under experiments/v3. Install its requirements in an isolated Python environment; replaying saved observations does not require new answering-model calls. The following commands use Windows PowerShell, matching the recorded environment and frozen path conventions.')
    for command in ['cd experiments/v3','python -m pip install -r requirements.txt','python -m unittest discover -s tests -p test_features_v3.py -v','python verify_v3.py','python train_v3.py evaluate']:
        p=paragraph(doc,command);p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(2)
        for run in p.runs:run.font.name='Consolas';run.font.size=Pt(8.5)
    paragraph(doc,'Repository https://github.com/thisisyk/cachescope-masc520')
    paragraph(doc,'The snapshot keeps earlier data and model artifacts unchanged. Creating a new experiment requires a separate version with fresh evaluation sources; overwriting the frozen selection would invalidate its recorded provenance. Report DOCX and PDF are supplied together, with editable prose, figures, and a builder under reports/source.')
    for element in list(doc.element.iter(qn('w:pBdr'))):element.getparent().remove(element)
    for element in list(doc.styles.element.iter(qn('w:pBdr'))):element.getparent().remove(element)
    core=doc.core_properties;core.title='CacheScope Neural Risk Prediction for Prompt Compression';core.subject='MASC 520 Final Project';core.author='';core.last_modified_by='';core.comments='';core.keywords='prompt compression; neural risk; selective prediction';core.created=datetime(2026,11,11,tzinfo=timezone.utc);core.modified=datetime(2026,11,11,tzinfo=timezone.utc)
    path=REPORTS/'CacheScope_MASC520_Final_Report.docx';doc.save(path)
    (SOURCE/'references.json').write_text(json.dumps(REFERENCES,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'docx':str(path),'source':str(SOURCE/'manuscript.md'),'figures':[str(flow),str(harm)]}))
if __name__=='__main__':main()
