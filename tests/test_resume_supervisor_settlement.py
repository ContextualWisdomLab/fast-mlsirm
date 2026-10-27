"""Inert real-process regression controls; no estimator import or execution.

A root forks one SIGTERM-resistant descendant in its new session. Both have a
finite fixture-only SIGALRM. The test records pre-recovery observations, then
signals only that retained group and checks absence. Grandchild absence is not
an exact grandchild-wait claim (the supervisor only owns the root's wait).
"""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(os.environ.get('SUPERVISOR_SOURCE',Path(__file__).resolve().parents[1]))
sp=importlib.util.spec_from_file_location('settlement_candidate',ROOT/'tools/resume_remaining_units.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
EVIDENCE=Path(os.environ.get('SUPERVISOR_EVIDENCE',os.environ['TMPDIR']))
POLICY=dict(m.DEFAULT_POLICY,poll_seconds=0.01,heartbeat_seconds=0.02,settlement_seconds=0.4,min_free_bytes=1)
FORK_SCRIPT='''import os,signal,sys,time
from pathlib import Path
signal.alarm(12)
pid=os.fork()
if pid==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 signal.alarm(12)
 sys.stderr.write('FMLSIRM_DIF_PROGRESS {"event":"optimizer_terminal"}\\n');sys.stderr.flush()
 Path(sys.argv[1]).write_text(str(os.getpid()))
 while True:time.sleep(.05)
while True:time.sleep(.05)
'''

class SettlementTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.root=Path(self.tmp.name)
  self.children=[];self.terminals=[];self.events=[]
  self.original_popen=m.subprocess.Popen;self.original_resources=m.group_resources
  self.original_killpg=os.killpg;self.original_absent=m.group_absent
 def tearDown(self):
  # Emergency recovery is logged separately, never credited as product cleanup.
  recovery=[]
  for p in self.children:
   try:
    if not self.original_absent(p.pid):
     self.original_killpg(p.pid,signal.SIGKILL);recovery.append({'pgid':p.pid,'signal':9})
   except ProcessLookupError:pass
   try:p.wait(timeout=3)
   except BaseException as e:recovery.append({'wait_error':repr(e)})
   end=time.monotonic()+3
   while not self.original_absent(p.pid) and time.monotonic()<end:time.sleep(.01)
   recovery.append({'pid':p.pid,'root_returncode':p.returncode,'group_absent':self.original_absent(p.pid)})
  reconcile=getattr(m,'reconcile_custody',None)
  if reconcile is not None:reconcile()
  payload={'case':self.id(),'source':str(ROOT),'terminals':self.terminals,'events':self.events,'recovery':recovery}
  (EVIDENCE/(self._testMethodName+'.json')).write_text(json.dumps(payload,indent=2)+'\n')
  self.tmp.cleanup()
 def popen(self,*a,**kw):
  p=self.original_popen(*a,**kw)
  if kw.get('start_new_session'):self.children.append(p)
  return p
 def ready(self,pgid):
  end=time.monotonic()+3
  while not (self.root/'ready').exists() and time.monotonic()<end:time.sleep(.005)
  self.assertTrue((self.root/'ready').exists(),'fixture descendant did not publish readiness')
  descendant=int((self.root/'ready').read_text());os.kill(descendant,0)
  self.assertEqual(os.getpgid(descendant),pgid)
  return descendant
 def invoke(self,script=None,**kw):
  args=[sys.executable,'-B','-c',FORK_SCRIPT,str(self.root/'ready')] if script is None else [sys.executable,'-B','-c',script]
  with patch.object(m.subprocess,'Popen',side_effect=self.popen):
   t=m.supervise(args,self.root/'process',dict(os.environ),POLICY,**kw)
  self.terminals.append(t);return t
 def observer_error(self,pgid):self.ready(pgid);raise RuntimeError('PRIMARY_OBSERVER_RESOURCE')
 def assert_retired(self,t):
  self.assertTrue(t['child_waited'],t)
  self.assertTrue(t['process_group_absent'],t)
  self.assertTrue(self.original_absent(t['pid']),t)
  self.assertFalse(m.successful_terminal(t))
  self.assertEqual(t['progress_records'],1,'complete descendant terminal bytes must be drained')
 def test_resource_error_settles_term_resistant_descendant(self):
  with patch.object(m,'group_resources',side_effect=self.observer_error):t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error']);self.assert_retired(t)
 def test_normal_root_exit_settles_same_descendant(self):
  def exit_root(pgid):
   self.ready(pgid);os.kill(pgid,signal.SIGTERM);return []
  with patch.object(m,'group_resources',side_effect=exit_root):t=self.invoke()
  self.assertIsNone(t['error']);self.assertEqual(t['stop_reason'],'descendant_survived');self.assert_retired(t)
 def test_primary_survives_group_signal_refusal(self):
  denied=[False]
  def signals(pgid,sig):
   self.events.append({'signal':sig,'pgid':pgid})
   if sig==signal.SIGTERM and not denied[0]:denied[0]=True;raise PermissionError('SECONDARY_SIGNAL_REFUSAL')
   return self.original_killpg(pgid,sig)
  with patch.object(m,'group_resources',side_effect=self.observer_error),patch.object(m.os,'killpg',side_effect=signals):t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error']);self.assert_retired(t)
  self.assertTrue(any('SECONDARY_SIGNAL_REFUSAL' in str(e) for e in t.get('settlement_errors',[])),t)
 def test_primary_survives_terminal_drain_fault_and_sibling_is_drained(self):
  original_fsync=m.os.fsync;active=[False]
  def observer(pgid):
   self.ready(pgid);active[0]=True;raise RuntimeError('PRIMARY_OBSERVER_RESOURCE')
  def fsync(fd):
   if active[0]:
    actual=os.fstat(fd);target=(self.root/'process/stdout').stat()
    if (actual.st_dev,actual.st_ino)==(target.st_dev,target.st_ino):raise OSError('SECONDARY_STDOUT_FSYNC')
   return original_fsync(fd)
  # Observer primary occurs after readiness; then only stdout drain fails.
  with patch.object(m,'group_resources',side_effect=observer),patch.object(m.os,'fsync',side_effect=fsync):t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error']);self.assert_retired(t)
  self.assertTrue(any('SECONDARY_STDOUT_FSYNC' in str(e) for e in t.get('settlement_errors',[])),t)
 def test_group_observation_and_wait_refusal_retain_custody(self):
  original_wait=subprocess.Popen.wait;fault=[False]
  def wait(p,*a,**kw):
   if p in self.children and not fault[0]:fault[0]=True;raise OSError('SECONDARY_WAIT_REFUSAL')
   return original_wait(p,*a,**kw)
  def absent(pgid):raise PermissionError('SECONDARY_GROUP_OBSERVATION')
  with patch.object(m,'group_resources',side_effect=self.observer_error),patch.object(m,'group_absent',side_effect=absent),patch.object(subprocess.Popen,'wait',wait):
   t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error'])
  self.assertEqual(t.get('custody_status'),'retained_no_automatic_retry',t)
  self.assertFalse(t['process_group_absent']);self.assertFalse(m.successful_terminal(t))
  self.assertTrue(any('SECONDARY_WAIT_REFUSAL' in str(e) for e in t.get('settlement_errors',[])),t)
  self.assertTrue(any('SECONDARY_GROUP_OBSERVATION' in str(e) for e in t.get('settlement_errors',[])),t)
  self.assertTrue(m.UNRESOLVED_CUSTODY)
  with self.assertRaisesRegex(RuntimeError,'unresolved custody'):self.invoke(script='pass')
  self.assertEqual(len(self.children),1,'unresolved ownership must forbid a second spawn')
  self.assertTrue(m.reconcile_custody());self.assertFalse(m.UNRESOLVED_CUSTODY)
 def test_initial_persistence_error_is_primary_and_descendant_is_settled(self):
  original=m.persist
  def persist(path,value,*a,**kw):
   if Path(path).name=='started.json':self.ready(self.children[-1].pid);raise OSError('PRIMARY_STARTED_PERSISTENCE')
   return original(path,value,*a,**kw)
  with patch.object(m,'persist',side_effect=persist):t=self.invoke()
  self.assertIn('PRIMARY_STARTED_PERSISTENCE',t['error']);self.assert_retired(t)
 def test_spawn_error_retains_receipt_and_does_not_invent_wait(self):
  with patch.object(m.subprocess,'Popen',side_effect=FileNotFoundError('PRIMARY_SPAWN')):
   t=m.supervise(['/no/inert/executable'],self.root/'process',dict(os.environ),POLICY)
  self.terminals.append(t);self.assertIn('PRIMARY_SPAWN',t['error']);self.assertFalse(t['child_waited'])
  self.assertFalse(m.successful_terminal(t));self.assertFalse(getattr(m,'UNRESOLVED_CUSTODY',{}))
 def test_inherited_sigchld_ignored_refuses_before_output_or_child(self):
  old=signal.signal(signal.SIGCHLD,signal.SIG_IGN)
  try:
   try:t=self.invoke(script='import sys;sys.exit(17)')
   except RuntimeError as e:self.assertIn('SIGCHLD',str(e))
   else:self.fail('unsafe SIGCHLD admitted a real child: '+str(t))
   self.assertFalse((self.root/'process').exists());self.assertEqual(self.children,[])
  finally:signal.signal(signal.SIGCHLD,old)
 def test_main_sigchld_guard_precedes_native_inspect_claim_and_output(self):
  old=signal.signal(signal.SIGCHLD,signal.SIG_IGN)
  try:
   with patch.object(m,'verify_spec',return_value={}),patch.object(m,'inspect',side_effect=AssertionError('native inspect must not run')) as inspect:
    with self.assertRaisesRegex(RuntimeError,'SIGCHLD'):
     m.main(['--binary',str(ROOT/'tools/resume_remaining_units.py'),'--spec',str(ROOT/'tools/resume_remaining_units.py'),'--spec-sha256','0'*64])
   self.assertEqual(inspect.call_count,0);self.assertEqual(self.children,[])
  finally:signal.signal(signal.SIGCHLD,old)

 def test_secondary_stream_close_failure_preserves_primary_and_sibling_close(self):
  primary=RuntimeError('PRIMARY_OBSERVER_RESOURCE');closed=[]
  def observer(pgid):self.ready(pgid);raise primary
  original_fdopen=m.os.fdopen
  class Stream:
   def __init__(self,wrapped,name):self.wrapped=wrapped;self.name=name
   def __getattr__(self,k):return getattr(self.wrapped,k)
   def close(self):
    closed.append(self.name);self.wrapped.close()
    if self.name=='stdout':raise OSError('SECONDARY_STDOUT_CLOSE')
  def fdopen(fd,mode,*a,**kw):
   f=original_fdopen(fd,mode,*a,**kw)
   if mode=='wb':
    actual=os.fstat(fd)
    for name in ('stdout','stderr'):
     path=self.root/'process'/name
     if path.exists():
      target=path.stat()
      if (actual.st_dev,actual.st_ino)==(target.st_dev,target.st_ino):return Stream(f,name)
   return f
  with patch.object(m,'group_resources',side_effect=observer),patch.object(m.os,'fdopen',side_effect=fdopen):t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error']);self.assert_retired(t)
  self.assertEqual(closed,['stdout','stderr']);self.assertTrue(any('SECONDARY_STDOUT_CLOSE' in str(e) for e in t['settlement_errors']))
 def test_two_independent_cleanup_faults_retained_with_original_primary(self):
  primary=RuntimeError('PRIMARY_OBSERVER_RESOURCE');denied=[False];original=m.persist
  def observer(pgid):self.ready(pgid);raise primary
  def signals(pgid,sig):
   if sig==signal.SIGTERM and not denied[0]:denied[0]=True;raise PermissionError('SECONDARY_SIGNAL_REFUSAL')
   return self.original_killpg(pgid,sig)
  def persist(path,value,*a,**kw):
   if Path(path).name=='terminal.json':
    self.terminals.append(dict(value));raise OSError('SECONDARY_TERMINAL_PERSISTENCE')
   return original(path,value,*a,**kw)
  with patch.object(m,'group_resources',side_effect=observer),patch.object(m.os,'killpg',side_effect=signals),patch.object(m,'persist',side_effect=persist):
   try:self.invoke()
   except RuntimeError as error:self.assertIs(error,primary);self.assertIsInstance(error.__cause__,OSError)
   else:self.fail('terminal persistence fault swallowed')
  self.assertTrue(any('SECONDARY_SIGNAL_REFUSAL' in str(e) for e in self.terminals[-1]['settlement_errors']))
  self.assertFalse(m.UNRESOLVED_CUSTODY)
 def test_main_real_child_error_settles_and_drains_before_lease_release(self):
  # Native acquisition/validation are inert adapters; the actual main owns the
  # real fixture canonical lease and calls the unchanged supervisor with forks.
  lease_path=self.root/'fixture-main-lease';lease_path.touch()
  admission_file=self.root/'admission';admission_file.write_text('{}')
  cancel=self.root/'cancel';claim=self.root/'claim';output=self.root/'main'
  identity={'path':str(lease_path),'inode':lease_path.stat().st_ino}
  admission={'claim_path':str(claim),'lease':identity}
  original_supervise=m.supervise;original_persist=m.persist;checks=[]
  def supervised(argv,directory,env,policy,**kw):
   # Substitute only the child argv; use the real returned Popen and group.
   return original_supervise([sys.executable,'-B','-c',FORK_SCRIPT,str(self.root/'ready')],directory,env,POLICY,**kw)
  def persist(path,value,*a,**kw):
   if Path(path)==output/'process/terminal.json':
    self.terminals.append(dict(value));self.assert_retired(value)
    with lease_path.open('r+') as contender:
     with self.assertRaises(BlockingIOError):m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
    checks.append('terminal_drained_lease_held')
   return original_persist(path,value,*a,**kw)
  args=['--binary',str(ROOT/'tools/resume_remaining_units.py'),'--spec',str(ROOT/'tools/resume_remaining_units.py'),'--spec-sha256','0'*64,'--execute','--output',str(output),'--admission',str(admission_file),'--cancel-file',str(cancel)]
  with patch.object(m,'verify_spec',return_value={}),patch.object(m,'validate_admission',return_value=admission),patch.object(m,'inspect',return_value={'remaining_items':[3],'item_ids':['a','b','c','d']}),patch.object(m,'supervise',side_effect=supervised),patch.object(m,'group_resources',side_effect=self.observer_error),patch.object(m,'persist',side_effect=persist),patch.object(m.subprocess,'Popen',side_effect=self.popen):
   self.assertEqual(m.main(args),2)
  self.assertEqual(checks,['terminal_drained_lease_held']);self.assertEqual(len(self.children),1)
  with lease_path.open('r+') as contender:m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
  self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'blocked_no_automatic_retry')
  self.assertFalse(m.UNRESOLVED_CUSTODY)
 def test_recovery_lease_is_held_until_group_absence_and_drain(self):
  lease_path=self.root/'fixture-lease';lease_path.touch();lease=lease_path.open('r+')
  m.fcntl.flock(lease,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
  identity={'path':str(lease_path),'inode':lease_path.stat().st_ino}
  original=m.persist;order=[]
  def persist(path,value,*a,**kw):
   if Path(path).name=='terminal.json':
    self.assertTrue(value['process_group_absent']);self.assertEqual(value['progress_records'],1)
    contender=lease_path.open('r+')
    try:
     with self.assertRaises(BlockingIOError):m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
    finally:contender.close()
    order.append('drained_terminal_while_lease_held')
   return original(path,value,*a,**kw)
  try:
   with patch.object(m,'group_resources',side_effect=self.observer_error),patch.object(m,'persist',side_effect=persist):
    t=self.invoke(lease_fd=lease.fileno(),lease_identity=identity)
   self.assert_retired(t);self.assertEqual(order,['drained_terminal_while_lease_held'])
  finally:lease.close()
  with lease_path.open('r+') as contender:m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
 def test_unresolved_duplicate_lease_survives_original_close_until_reconcile(self):
  lease_path=self.root/'fixture-lease';lease_path.touch();lease=lease_path.open('r+')
  m.fcntl.flock(lease,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
  identity={'path':str(lease_path),'inode':lease_path.stat().st_ino}
  try:
   with patch.object(m,'group_resources',side_effect=self.observer_error),patch.object(m,'group_absent',side_effect=PermissionError('OBSERVATION_UNKNOWN')):
    t=self.invoke(lease_fd=lease.fileno(),lease_identity=identity)
   lease.close();self.assertTrue(t['lease_retained']);self.assertTrue(m.UNRESOLVED_CUSTODY)
   with lease_path.open('r+') as contender:
    with self.assertRaises(BlockingIOError):m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
   # Allow OS orphan reaping to become observable without another signal.
   end=time.monotonic()+2
   while not self.original_absent(t['pid']) and time.monotonic()<end:time.sleep(.01)
   self.assertTrue(m.reconcile_custody());self.assertFalse(m.UNRESOLVED_CUSTODY)
   with lease_path.open('r+') as contender:m.fcntl.flock(contender,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
  finally:
   if not lease.closed:lease.close()
 def test_group_wait_interruption_preserves_primary_and_still_escalates(self):
  original_sleep=m.time.sleep;in_cleanup=[False];injected=[False]
  def observer(pgid):
   self.ready(pgid);in_cleanup[0]=True;raise RuntimeError('PRIMARY_OBSERVER_RESOURCE')
  def sleep(seconds):
   if in_cleanup[0] and not injected[0]:injected[0]=True;raise InterruptedError('SECONDARY_GROUP_WAIT')
   return original_sleep(seconds)
  with patch.object(m,'group_resources',side_effect=observer),patch.object(m.time,'sleep',side_effect=sleep):t=self.invoke()
  self.assertIn('PRIMARY_OBSERVER_RESOURCE',t['error']);self.assert_retired(t)
  self.assertTrue(any('SECONDARY_GROUP_WAIT' in str(e) for e in t.get('settlement_errors',[])),t)
 def test_terminal_receipt_error_keeps_original_primary_object_after_settlement(self):
  primary=RuntimeError('PRIMARY_IDENTITY');secondary=OSError('SECONDARY_TERMINAL_PERSISTENCE');original=m.persist
  def observer(pgid):self.ready(pgid);raise primary
  def persist(path,value,*a,**kw):
   if Path(path).name=='terminal.json':
    self.terminals.append(dict(value));self.assert_retired(value);raise secondary
   return original(path,value,*a,**kw)
  with patch.object(m,'group_resources',side_effect=observer),patch.object(m,'persist',side_effect=persist):
   try:self.invoke()
   except RuntimeError as e:self.assertIs(e,primary);self.assertIs(e.__cause__,secondary)
   else:self.fail('terminal receipt failure swallowed')
  self.assertFalse(m.UNRESOLVED_CUSTODY)
 def test_terminal_receipt_error_without_body_primary_is_not_success(self):
  secondary=OSError('SECONDARY_ONLY_TERMINAL_PERSISTENCE');original=m.persist
  def persist(path,value,*a,**kw):
   if Path(path).name=='terminal.json':self.terminals.append(dict(value));raise secondary
   return original(path,value,*a,**kw)
  with patch.object(m,'persist',side_effect=persist):
   try:self.invoke(script='pass')
   except OSError as e:self.assertIs(e,secondary)
   else:self.fail('terminal receipt failure swallowed')
  self.assertTrue(self.terminals[-1]['child_waited']);self.assertTrue(self.terminals[-1]['process_group_absent'])
 def test_final_combined_stream_cap_is_preserved_with_split_drains(self):
  script='import sys;sys.stdout.write("x"*60);sys.stderr.write("y"*60)'
  with patch.dict(POLICY,max_stream_bytes=100):t=self.invoke(script=script)
  self.assertEqual(t['stop_reason'],'stream_limit');self.assertTrue(t['drain_truncated']);self.assertFalse(m.successful_terminal(t))
 def test_root_nonzero_status_is_preserved_under_safe_sigchld(self):
  t=self.invoke(script='import sys;sys.exit(17)')
  self.assertEqual(t['child_exit'],17);self.assertTrue(t['child_waited']);self.assertTrue(t['process_group_absent']);self.assertFalse(m.successful_terminal(t))
 def test_inherited_custom_sigchld_handler_is_not_silently_changed(self):
  def handler(signum,frame):pass
  old=signal.signal(signal.SIGCHLD,handler)
  try:
   with self.assertRaisesRegex(RuntimeError,'SIGCHLD'):self.invoke(script='pass')
   self.assertIs(signal.getsignal(signal.SIGCHLD),handler);self.assertEqual(self.children,[])
  finally:signal.signal(signal.SIGCHLD,old)

if __name__=='__main__':unittest.main(verbosity=2)
