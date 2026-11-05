"""Process original IXI pairs with subject IDs and resumable per-case records."""
import csv
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import SimpleITK as sitk
from register_ixi_t1_to_t2 import register, index_files, original_pair_key

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'IXI_registered_original'

def run_one(task):
    subject,t1,t2=task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    dest=OUT/subject
    dest.mkdir(exist_ok=True)
    record=dest/'registration.json'
    if record.exists():
        try:
            previous=json.loads(record.read_text(encoding='utf-8'))
            if (dest/'T1_registered.nii.gz').exists():
                return subject,previous
        except (json.JSONDecodeError,OSError):
            pass
    start=time.time()
    staged_t1=dest/'_input_t1.nii.gz'
    staged_t2=dest/'_input_t2.nii.gz'
    try:
        # Older Windows ITK builds cannot reliably open Unicode source paths.
        # Binary-copy inputs to ASCII filenames, then retain original provenance.
        shutil.copyfile(t1,staged_t1)
        shutil.copyfile(t2,staged_t2)
        result=register(staged_t1,staged_t2,dest,compact=True)
        result['t1']=t1
        result['t2']=t2
        result['subject']=subject
        result['elapsed_seconds']=time.time()-start
        result['status']='awaiting_anatomical_QC'
        record.write_text(json.dumps(result,indent=2),encoding='utf-8')
        (dest/'error.json').unlink(missing_ok=True)
        return subject,result
    except Exception as exc:
        result={'subject':subject,'status':'failed','error':repr(exc),'t1':t1,'t2':t2}
        (dest/'error.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return subject,result
    finally:
        staged_t1.unlink(missing_ok=True)
        staged_t2.unlink(missing_ok=True)

def main():
    OUT.mkdir(exist_ok=True)
    a=index_files(r'F:\IXI\IXI数据集\IXI-T1','T1')
    b=index_files(r'F:\IXI\IXI数据集\IXI-T2','T2')
    tasks=[]
    for number in sorted(a.keys()&b.keys()):
        ka,kb=original_pair_key(a[number]),original_pair_key(b[number])
        if ka is None or ka!=kb:
            continue
        tasks.append((ka,str(a[number]),str(b[number])))
    with (OUT/'paired_subjects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject','t1','t2']);writer.writerows(tasks)
    print(f'Processing {len(tasks)} original-ID pairs with 6 workers',flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(run_one,task) for task in tasks]
        for index,future in enumerate(as_completed(futures),1):
            subject,result=future.result()
            qc=result.get('quality_control',{})
            print(index,subject,result['status'],qc.get('decision'),qc.get('candidate',{}).get('fixed_mask_nmi'),flush=True)
    print('BATCH_COMPLETE',flush=True)

if __name__=='__main__':
    main()
