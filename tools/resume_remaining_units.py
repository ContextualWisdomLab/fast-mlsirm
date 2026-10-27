"""Trusted-private missing-only launcher. Default mode restores and plans only.

Scientific children have NO elapsed-time termination. Tool wait windows are not
native deadlines. Stops are explicit cancellation, process errors, malformed
progress, lease loss, output/RSS/disk limits, or source/input binding changes.
Every stop waits for the actual child, drains terminal bytes and preserves state.
The parent alone supplies reviewed execution authority; this is not hostile-UID
execution authentication. A stable selected-final unit needs separate admission.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

UNITS=('alternative','pair','family')
DEFAULT_POLICY={'max_rss_bytes':8*1024**3,'max_stream_bytes':128*1024**2,
 'min_free_bytes':1024**3,'poll_seconds':1.0,'heartbeat_seconds':5.0,'settlement_seconds':10.0}
EVENTS={'optimizer_start','optimizer_terminal','fit_start','resume_unit_start','resume_unit_terminal'}

def validate_policy(policy):
    if set(policy)!=set(DEFAULT_POLICY):raise ValueError('explicit resource/settlement policy required; wallclock deadlines are not supported')
    for key,value in policy.items():
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('resource and observation limits must be finite positive numbers')
    for key in ('max_rss_bytes','max_stream_bytes','min_free_bytes'):
        if type(policy[key]) is not int:raise ValueError('byte bounds must be integers')
    return dict(policy)

def stop_for_elapsed(elapsed,policy):
    """Elapsed duration is observable, never a reason to kill scientific work."""
    return False

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

def load(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>32*1024**2:
        raise ValueError('private JSON type/size refused')
    return json.loads(path.read_text(),object_pairs_hook=unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def persist(path,value,replace=False):
    path=Path(path);temp=path.with_suffix('.pending')
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if replace:os.replace(temp,path)
    else:os.link(temp,path);temp.unlink()
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def reference(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('regular nonsymlink input required')
    return {'path':str(path.resolve(strict=True)),'sha256':digest(path)}

def verify_reference(ref):
    p=Path(ref['path'])
    if reference(p)!=ref:raise ValueError('admission reference mismatch')
    return p

def verify_spec(spec,expected):
    if digest(spec)!=expected:raise ValueError('external resume spec digest mismatch')
    value=load(spec)
    for ref in [value['journal'],*value['states'],*value['provenance'],value['checkpoint_authority']]:verify_reference(ref)
    return value

def validate_admission(admission,bindings,policy,unit):
    if admission.get('schema')!=2 or admission.get('decision')!='admit_exact_remaining_units' or admission.get('enabled') is not True:
        raise ValueError('parent execution admission required')
    if admission.get('bindings')!=bindings or admission.get('policy')!=policy:
        raise ValueError('parent admission generation/resource policy mismatch')
    allowed=admission.get('allowed_units')
    if not isinstance(allowed,list) or len(allowed)!=len(set(allowed)) or unit not in allowed or any(x not in (*UNITS,'final') for x in allowed):
        raise ValueError('parent unit scope mismatch')
    for key in ('checkpoint_verdict','resume_consumer_verdict'):
        p=verify_reference(admission[key])
        if not p.stat().st_size:raise ValueError('empty independent verdict')
    if type(admission.get('claim_path')) is not str or not Path(admission['claim_path']).is_absolute():raise ValueError('absolute exclusive claim path required')
    if admission.get('cancel_file')!=bindings.get('cancel_file'):raise ValueError('cancel control mismatch')
    if unit=='alternative':
        l=admission['lease'];p=Path(l['path'])
        if p.is_symlink() or not p.is_file() or p.stat().st_ino!=l['inode']:raise ValueError('canonical lease inode mismatch')
    return admission

def inspect(binary,spec,expected):
    env=dict(os.environ,FMLS_DS_GPU_LEASED='0',FAST_MLSIRM_DIF_PROGRESS='0')
    env.pop('FMLS_RESUME_ADMITTED_SPEC',None);env.pop('FMLS_RESUME_ADMITTED_UNIT',None)
    # This short timeout bounds the restore-only tool, not an optimizing child.
    q=subprocess.run([str(binary),'inspect',str(spec),expected],capture_output=True,text=True,env=env,timeout=60,check=False)
    if q.returncode!=0:raise ValueError('native restore-only admission refused: '+q.stderr.strip())
    a=json.loads(q.stdout)
    if a.get('fit_launches')!=0 or a.get('remaining_items')!=[3] or [p['item_index'] for p in a.get('completed',[])]!=[0,1,2]:
        raise ValueError('unexpected missing-only scope')
    return a

def successful_terminal(t):
    return (type(t.get('child_exit')) is int and t['child_exit']==0
        and t.get('child_waited') is True and t.get('process_group_absent') is True
        and t.get('stop_reason') is None and t.get('error') is None
        and not t.get('drain_truncated') and not t.get('settlement_errors') and not t.get('progress_errors') and not any(t.get('partial_tail_bytes',{}).values()))

def group_absent(pgid):
    try:os.killpg(pgid,0);return False
    except ProcessLookupError:return True
    except PermissionError:return False  # unknown authority is never evidence of absence

def group_resources(pgid):
    raw=subprocess.check_output(['ps','-axo','pid=,pgid=,rss=,pcpu=,state='],text=True,timeout=10)
    rows=[]
    for line in raw.splitlines():
        fields=line.split()
        if len(fields)==5 and int(fields[1])==pgid:
            rows.append({'pid':int(fields[0]),'rss_bytes':int(fields[2])*1024,'cpu_percent':float(fields[3]),'state':fields[4]})
    return rows

def supervise(argv,output,env,policy,*,lease_fd=None,lease_identity=None,cancel_file=None):
    """One actual child, unlimited scientific lifetime, bounded stop/settlement.

    Raw files are flushed/fsynced by this observer each poll. Numerical progress
    comes from native events; a heartbeat is explicitly only observer liveness.
    No automatic restart occurs, including after an uncertain settlement.
    """
    policy=validate_policy(policy);output=Path(output);output.mkdir(mode=0o700,exist_ok=False)
    start=time.monotonic();t={'schema':2,'argv':argv,'started':datetime.datetime.now().astimezone().isoformat(),
      'host':socket.gethostname(),'policy':policy,'deadline_seconds':None,'child_waited':False,
      'child_exit':None,'process_group_absent':False,'stop_reason':None,'error':None,
      'progress_errors':[],'progress_records':0,'partial_tail_bytes':{'stdout':0,'stderr':0},'lease':lease_identity}
    p=None;files={};offsets={'stdout':0,'stderr':0};buffers={'stdout':b'','stderr':b''};last_heartbeat=0
    def drain():
        total=0
        for name,f in files.items():
            f.flush();os.fsync(f.fileno())
            size=os.fstat(f.fileno()).st_size;total+=size
            if total>policy['max_stream_bytes']:stop('stream_limit')
            with (output/name).open('rb') as src:
                src.seek(offsets[name])
                while offsets[name]<size:
                    allowed=policy['max_stream_bytes']-offsets[name]
                    if allowed<=0:
                        t['drain_truncated']=True;offsets[name]=size;break
                    b=src.read(min(65536,size-offsets[name],allowed));offsets[name]+=len(b)
                    buf=buffers[name]+b;lines=buf.split(b'\n');buffers[name]=lines.pop()
                    for line in lines:
                        if len(line)>8*1024**2:raise ValueError('progress record cap exceeded')
                        if name=='stderr' and line.startswith(b'FMLSIRM_DIF_PROGRESS '):
                            try:
                                e=json.loads(line[len(b'FMLSIRM_DIF_PROGRESS '):])
                                if e.get('event') not in EVENTS:raise ValueError('unknown event')
                                t['progress_records']+=1;t['last_progress_event']=e
                            except (ValueError,UnicodeError,AttributeError) as e:t['progress_errors'].append(str(e))
                        elif line:t['progress_errors'].append('unexpected '+name+' line') if name=='stderr' else None
                    if len(buffers[name])>8*1024**2:raise ValueError('progress record cap exceeded')
        return total
    def stop(reason):
        if t['stop_reason'] is None:t['stop_reason']=reason
        if p is not None and p.poll() is None and not group_absent(p.pid):
            try:os.killpg(p.pid,signal.SIGTERM)
            except PermissionError as error:
                t.setdefault('settlement_errors',[]).append('group signal refused: '+str(error))
                p.terminate()
    previous_handlers={}
    def interrupted(signum,frame):raise InterruptedError('explicit supervisor signal '+str(signum))
    if __import__('threading').current_thread() is __import__('threading').main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[signum]=signal.signal(signum,interrupted)
    try:
        for name in ('stdout','stderr'):
            fd=os.open(output/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);files[name]=os.fdopen(fd,'wb')
        p=subprocess.Popen(argv,env=env,stdout=files['stdout'],stderr=files['stderr'],start_new_session=True,
            pass_fds=() if lease_fd is None else (lease_fd,))
        t.update(pid=p.pid,pgid=p.pid)
        persist(output/'started.json',t)
        # Immediate fsynced custody precedes the first wait.
        persist(output/'heartbeat.json',dict(t,observer_running=True,elapsed_seconds=0))
        while p.poll() is None:
            total=drain();resources=group_resources(p.pid)
            t['resources']=resources;t['elapsed_seconds']=time.monotonic()-start
            if sum(r['rss_bytes'] for r in resources)>policy['max_rss_bytes']:stop('rss_limit')
            if total>policy['max_stream_bytes']:stop('stream_limit')
            if shutil.disk_usage(output).free<policy['min_free_bytes']:stop('disk_limit')
            if t['progress_errors']:stop('progress_error')
            if cancel_file is not None and Path(cancel_file).exists():stop('explicit_cancel')
            if lease_fd is not None:
                fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                current=os.fstat(lease_fd);named=os.stat(lease_identity['path'],follow_symlinks=False)
                if (current.st_dev,current.st_ino)!=(named.st_dev,named.st_ino):stop('lease_identity_lost')
            if time.monotonic()-last_heartbeat>=policy['heartbeat_seconds']:
                persist(output/'heartbeat.json',dict(t,observer_running=True),replace=True);last_heartbeat=time.monotonic()
            if t['stop_reason'] is not None:break
            time.sleep(policy['poll_seconds'])
        if t['stop_reason'] is not None:
            try:p.wait(timeout=policy['settlement_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=policy['settlement_seconds'])
        else:p.wait()
        t['child_waited']=True;t['child_exit']=p.returncode
        if not group_absent(p.pid):
            stop('descendant_survived')
            try:os.killpg(p.pid,signal.SIGTERM)
            except PermissionError as error:t.setdefault('settlement_errors',[]).append(str(error))
            except ProcessLookupError:pass
            limit=time.monotonic()+policy['settlement_seconds']
            while not group_absent(p.pid) and time.monotonic()<limit:time.sleep(0.01)
            if not group_absent(p.pid):
                os.killpg(p.pid,signal.SIGKILL);limit=time.monotonic()+policy['settlement_seconds']
                while not group_absent(p.pid) and time.monotonic()<limit:time.sleep(0.01)
        # Terminal drain is after real wait and checks the entire final tail.
        if drain()>policy['max_stream_bytes'] and t['stop_reason'] is None:t['stop_reason']='stream_limit'
        if t['progress_errors'] and t['stop_reason'] is None:t['stop_reason']='progress_error'
    except BaseException as error:
        t['error']=type(error).__name__+': '+str(error)
        if p is not None:
            try:
                stop('observer_error');p.wait(timeout=policy['settlement_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=policy['settlement_seconds'])
            t['child_waited']=True;t['child_exit']=p.returncode
            try:drain()
            except BaseException as e:t['progress_errors'].append('terminal drain: '+str(e))
    finally:
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
        for f in files.values():f.close()
        t['partial_tail_bytes']={k:len(v) for k,v in buffers.items()}
        if any(t['partial_tail_bytes'].values()) and t['stop_reason'] is None:t['stop_reason']='partial_terminal_tail'
        t['elapsed_seconds']=time.monotonic()-start
        t['process_group_absent']=p is not None and group_absent(p.pid)
        t['streams']={name:reference(output/name) for name in files}
        persist(output/'terminal.json',t)
    return t

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha256',required=True);p.add_argument('--policy',type=Path)
    p.add_argument('--admission',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--unit',choices=(*UNITS,'final'),default='alternative')
    p.add_argument('--alternative-state',type=Path);p.add_argument('--pair-result',type=Path);p.add_argument('--family-result',type=Path)
    p.add_argument('--cancel-file',type=Path);p.add_argument('--execute',action='store_true')
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.execute and (a.admission is None or a.output is None):raise ValueError('execution needs parent admission and new output')
    if a.execute and a.cancel_file is None:raise ValueError('explicit cancellation path required')
    if a.cancel_file is not None and a.cancel_file.exists():raise ValueError('explicit cancellation already requested')
    policy=validate_policy(load(a.policy) if a.policy else DEFAULT_POLICY)
    bindings={'binary':reference(a.binary),'spec':reference(a.spec),'launcher':reference(__file__)}
    if a.output is not None:bindings['output']={'path':str(a.output.absolute())}
    for key in ('alternative_state','pair_result','family_result'):
        if (v:=getattr(a,key)) is not None:bindings[key]=reference(v)
    if a.cancel_file is not None:bindings['cancel_file']={'path':str(a.cancel_file.absolute())}
    verify_spec(a.spec,a.spec_sha256)
    admission=validate_admission(load(a.admission),bindings,policy,a.unit) if a.execute else None
    state=inspect(a.binary,a.spec,a.spec_sha256)
    plan={'schema':2,'bindings':bindings,'policy':policy,'allowed_units':list(UNITS),'enabled':False,
      'remaining_items':state['remaining_items'],'item_ids':state['item_ids'],'fit_launches':0,
      'initial_fit_launches':0,'completed_pair_refits':0,'selected_final_admitted':False,'native_deadline_seconds':None}
    if not a.execute:print(json.dumps(plan,indent=2));return 0
    if a.unit in ('pair','family','final') and a.alternative_state is None:raise ValueError('authentic alternative required')
    if a.unit in ('family','final') and a.pair_result is None:raise ValueError('complete missing pair required')
    if a.unit=='final' and a.family_result is None:raise ValueError('complete stable family required')
    a.output.mkdir(mode=0o700,exist_ok=False)
    # New owner directory itself excludes duplicate runs of this admission.
    claim=Path(admission['claim_path'])
    fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(str(a.output.resolve())+'\n');f.flush();os.fsync(f.fileno())
    lease=None;manifest={'schema':2,'unit':a.unit,'status':'running','bindings':bindings,'policy':policy,
      'admission':reference(a.admission),'claim_path':str(claim),'fit_launches_authorized':int(a.unit=='alternative')}
    persist(a.output/'manifest.json',manifest)
    try:
        identity=None
        if a.unit=='alternative':
            identity=admission['lease'];p=Path(identity['path'])
            fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW);lease=os.fdopen(fd,'r+')
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_ino!=identity['inode']:raise ValueError('canonical lease changed')
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            probe=subprocess.run([sys.executable,'-c','import fcntl,sys; f=open(sys.argv[1],"r+");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(73)',str(p)],timeout=10,check=False)
            if probe.returncode!=73:raise ValueError('independent canonical lease contender did not refuse')
        for key,ref in bindings.items():
            if key not in ('cancel_file','output'):verify_reference(ref)
        if shutil.disk_usage(a.output).free<policy['min_free_bytes']:raise ValueError('prelaunch disk admission failed')
        verify_spec(a.spec,a.spec_sha256)
        env=dict(os.environ,FAST_MLSIRM_DIF_PROGRESS='1',FMLS_DS_GPU_LEASED='1' if lease else '0',
          FMLS_RESUME_ADMITTED_SPEC=a.spec_sha256,FMLS_RESUME_ADMITTED_UNIT=a.unit)
        for key,name in (('alternative_state','ALTERNATIVE'),('pair_result','PAIR'),('family_result','FAMILY')):
            if key in bindings:env['FMLS_RESUME_ADMITTED_'+name+'_SHA256']=bindings[key]['sha256']
        args=[str(a.binary),a.unit,str(a.spec),a.spec_sha256]
        if a.unit in ('alternative','pair'):args+=['3']
        if a.unit=='pair':args+=[str(a.alternative_state)]
        if a.unit in ('family','final'):args+=[str(a.pair_result),str(a.alternative_state)]
        if a.unit=='final':args+=[str(a.family_result)]
        args+=[str(a.output/'native')]
        terminal=supervise(args,a.output/'process',env,policy,lease_fd=None if lease is None else lease.fileno(),lease_identity=identity,cancel_file=a.cancel_file)
        manifest['terminal']=reference(a.output/'process/terminal.json')
        manifest['status']='complete' if successful_terminal(terminal) else 'blocked_no_automatic_retry'
        if successful_terminal(terminal):
            manifest['result']=reference(a.output/'native/result.json')
            result=load(a.output/'native/result.json')
            status=(result['result']['status'] if a.unit=='pair' else result['family']['status'] if a.unit=='family' else result.get('status'))
            manifest['native_status']=status
            if a.unit=='pair' and status!='complete':manifest['status']='scientific_pair_incomplete'
            if a.unit=='family' and status!='stable_no_further_removal':manifest['status']='round_limit_incomplete'
            if a.unit=='alternative' and (result['fit']['converged'] is not True or result['fit']['dispatch_complete'] is not True):manifest['status']='alternative_not_admitted'
        persist(a.output/'manifest.json',manifest,replace=True)
        return 0 if manifest['status']=='complete' else 2
    except BaseException as e:
        manifest['status']='blocked_no_automatic_retry';manifest['error']=type(e).__name__+': '+str(e)
        persist(a.output/'manifest.json',manifest,replace=True);raise
    finally:
        if lease is not None:lease.close()

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as e:
        print('RESUME_LAUNCH_REFUSAL '+str(e),file=sys.stderr);raise SystemExit(2)
