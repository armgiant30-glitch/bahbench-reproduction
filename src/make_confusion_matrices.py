from pathlib import Path
import os
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ.get('BAHBENCH_RESULTS', 'results'))
OUT=ROOT/'confusion_matrices'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
MODELS=['facebook/hubert-base-ls960','facebook/wav2vec2-base','facebook/wav2vec2-large']
SHORT={'facebook/hubert-base-ls960':'HuBERT-base','facebook/wav2vec2-base':'wav2vec2-base','facebook/wav2vec2-large':'wav2vec2-large'}
TASKS=['ICBHI','TORGO','SEP-28k']

def find_json(folder, model):
    for p in glob.glob(str(folder/'*.json')):
        try: d=json.load(open(p,encoding='utf-8'))
        except Exception: continue
        a=d.get('args',{})
        if a.get('model')==model and a.get('mode') in (None,'frozen'):
            return d,p
    return None,None

def binary_cm(metrics):
    pc={str(x['label']):x for x in metrics['per_class']}
    n0=int(pc['0']['n']); n1=int(pc['1']['n'])
    r0=float(pc['0']['recall']); r1=float(pc['1']['recall'])
    tn=int(round(r0*n0)); tp=int(round(r1*n1)); fn=n1-tp; fp=n0-tn
    return np.array([[tn,fp],[fn,tp]],dtype=int),['0','1']

def multiclass_cm(d):
    pred=d.get('predictions')
    classes=[str(x) for x in d.get('classes',[])]
    if not pred: raise RuntimeError('missing predictions')
    yt=[str(x) for x in pred['y_true']]; yp=[str(x) for x in pred['y_pred']]
    idx={c:i for i,c in enumerate(classes)}; cm=np.zeros((len(classes),len(classes)),dtype=int)
    for a,b in zip(yt,yp): cm[idx[a],idx[b]]+=1
    return cm,classes

fig,axes=plt.subplots(3,3,figsize=(18,16),dpi=180)
fig.suptitle('BAHBench Paper Transformer Confusion Matrices',fontsize=18)
for i,task in enumerate(TASKS):
    folder=ROOT/('paper_transformer_'+('icbhi_cm' if task=='ICBHI' else 'torgo' if task=='TORGO' else 'sep28k'))
    for j,model in enumerate(MODELS):
        ax=axes[i,j]; d,p=find_json(folder,model)
        if d is None:
            ax.axis('off'); ax.set_title(f'{task} / {SHORT[model]}\nmissing'); continue
        try:
            cm,labels = multiclass_cm(d) if task=='ICBHI' else binary_cm(d['metrics'])
        except Exception as e:
            ax.axis('off'); ax.set_title(f'{task} / {SHORT[model]}\n{type(e).__name__}'); continue
        rs=cm.sum(axis=1,keepdims=True); norm=np.divide(cm,rs,out=np.zeros_like(cm,dtype=float),where=rs!=0)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_title(f'{task} / {SHORT[model]}',fontsize=10)
        ax.set_xticks(range(len(labels)),labels,fontsize=8); ax.set_yticks(range(len(labels)),labels,fontsize=8)
        ax.set_xlabel('Predicted',fontsize=8); ax.set_ylabel('True',fontsize=8)
        for a in range(len(labels)):
            for b in range(len(labels)):
                txt=f'{cm[a,b]}\n{norm[a,b]*100:.1f}%'
                ax.text(b,a,txt,ha='center',va='center',fontsize=7 if len(labels)>3 else 10,color='white' if norm[a,b]>0.5 else 'black')
        # individual save
        safe=SHORT[model].replace('/','_').replace(' ','_')
        fig2,ax2=plt.subplots(figsize=(5.5,4.8),dpi=200)
        ax2.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax2.set_title(f'{task} / {SHORT[model]} normalized confusion matrix')
        ax2.set_xticks(range(len(labels)),labels,fontsize=8); ax2.set_yticks(range(len(labels)),labels,fontsize=8)
        ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
        for a in range(len(labels)):
            for b in range(len(labels)): ax2.text(b,a,f'{cm[a,b]}\n{norm[a,b]*100:.1f}%',ha='center',va='center',fontsize=7 if len(labels)>3 else 10,color='white' if norm[a,b]>0.5 else 'black')
        fig2.tight_layout(); fig2.savefig(OUT/f'cm_{task.lower().replace("-","")}_{safe}.png',bbox_inches='tight'); plt.close(fig2)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/'confusion_matrix_grid.png',bbox_inches='tight'); plt.close(fig)
print('confusion matrices written to',OUT)
