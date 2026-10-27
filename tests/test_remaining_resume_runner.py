"""Pure/native-process admission controls; no estimator launch."""
from pathlib import Path
import importlib.util, os, sys, tempfile, unittest
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'tools/resume_remaining_units.py'
spec=importlib.util.spec_from_file_location('resume_remaining_units',p)
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NoWallclockCutoff(unittest.TestCase):
 def test_progressing_work_has_no_artificial_900s_budget(self):
  policy=m.validate_policy({'max_rss_bytes':1024**3,'max_stream_bytes':1048576,
   'min_free_bytes':1048576,'poll_seconds':0.01,'heartbeat_seconds':0.02,'settlement_seconds':2})
  self.assertNotIn('deadline_seconds',policy)
  self.assertFalse(m.stop_for_elapsed(901,policy))
  self.assertFalse(m.stop_for_elapsed(900000,policy))
 def test_deadline_configuration_is_rejected(self):
  with self.assertRaises(ValueError):m.validate_policy({'deadline_seconds':900})
 def test_resource_policy_is_explicit_and_finite(self):
  for value in (0,-1,True,float('nan'),float('inf')):
   with self.subTest(value=value):
    with self.assertRaises(ValueError):m.validate_policy(dict(m.DEFAULT_POLICY,max_rss_bytes=value))

class RunnerContracts(unittest.TestCase):
 def test_execute_without_parent_admission_has_zero_native_calls(self):
  args=['--binary',str(p),'--spec',str(p),'--spec-sha256','0'*64,'--execute']
  with patch.object(m,'inspect',side_effect=AssertionError('native forbidden')):
   with self.assertRaisesRegex(ValueError,'parent admission'):m.main(args)
 def test_durable_nonreplacement_permissions(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   f=Path(d)/'receipt.json';m.persist(f,{'x':1})
   self.assertEqual(f.stat().st_mode&0o777,0o600)
   with self.assertRaises(FileExistsError):m.persist(f,{'x':2})
   self.assertEqual(m.load(f),{'x':1})
 def test_terminal_cannot_promote_failed_or_unwaited(self):
  t={'child_exit':0,'child_waited':True,'process_group_absent':True,'stop_reason':None,
   'error':None,'progress_errors':[],'partial_tail_bytes':{'stderr':0,'stdout':0}}
  self.assertTrue(m.successful_terminal(t))
  for change in ({'stop_reason':'resource_limit'},{'child_waited':False},
   {'process_group_absent':False},{'error':'unknown'},{'child_exit':True},
   {'child_exit':-15},{'progress_errors':['bad']},{'partial_tail_bytes':{'stderr':1}}):
   self.assertFalse(m.successful_terminal(dict(t,**change)))
 def test_settled_valid_tail_and_error_tail(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   script='import sys; print(\'FMLSIRM_DIF_PROGRESS {"event":"optimizer_start","start":0}\',file=sys.stderr,flush=True)'
   t=m.supervise([sys.executable,'-c',script],Path(d)/'positive',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.01,heartbeat_seconds=0.02))
   self.assertTrue(m.successful_terminal(t),t);self.assertEqual(t['progress_records'],1)
   t=m.supervise([sys.executable,'-c','import sys; print(\'FMLSIRM_DIF_PROGRESS {"event":"foreign"}\',file=sys.stderr)'],Path(d)/'bad',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.01,heartbeat_seconds=0.02))
   self.assertFalse(m.successful_terminal(t));self.assertTrue(t['progress_errors'])
 def test_resource_stop_waits_and_keeps_outputs(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   t=m.supervise([sys.executable,'-c','import time; print("x"*20000,flush=True); time.sleep(5)'],Path(d)/'resource',dict(os.environ),dict(m.DEFAULT_POLICY,max_stream_bytes=100,poll_seconds=0.01))
   self.assertEqual(t['stop_reason'],'stream_limit');self.assertTrue(t['child_waited']);self.assertTrue(t['process_group_absent'])
 def test_partial_tail_refuses_without_erasing_complete_records(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   t=m.supervise([sys.executable,'-c','import sys; sys.stderr.write(\'FMLSIRM_DIF_PROGRESS {"event":"optimizer_start"}\\npartial\')'],Path(d)/'partial',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.01))
   self.assertFalse(m.successful_terminal(t));self.assertEqual(t['progress_records'],1);self.assertEqual(t['partial_tail_bytes']['stderr'],7)
 def test_actual_supervisor_does_not_kill_at_simulated_elapsed_901(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   original=m.time.monotonic;first=[True]
   def clock():
    if first[0]:first[0]=False;return original()-901
    return original()
   script='import sys,time;time.sleep(0.1);print(\'FMLSIRM_DIF_PROGRESS {"event":"optimizer_terminal"}\',file=sys.stderr,flush=True)'
   with patch.object(m.time,'monotonic',side_effect=clock):
    t=m.supervise([sys.executable,'-c',script],Path(d)/'progressing',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.01))
   self.assertGreater(t['elapsed_seconds'],900);self.assertTrue(m.successful_terminal(t),t)
 def test_group_permission_refusal_is_unknown_not_absent(self):
  with patch.object(m.os,'killpg',side_effect=PermissionError('inert observer denial')):
   self.assertFalse(m.group_absent(123456))
 def test_admission_generation_and_final_scope(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   v=Path(d)/'verdict';v.write_text('inert test verdict, not review acceptance')
   ref=m.reference(v);b={'binary':{'path':'inert','sha256':'one'}}
   a={'schema':2,'decision':'admit_exact_remaining_units','enabled':True,'bindings':b,
    'policy':m.DEFAULT_POLICY,'allowed_units':['pair'],'checkpoint_verdict':ref,
    'resume_consumer_verdict':ref,'claim_path':str(Path(d)/'claim'),'cancel_file':None}
   self.assertEqual(m.validate_admission(a,b,m.DEFAULT_POLICY,'pair'),a)
   with self.assertRaises(ValueError):m.validate_admission(a,{'changed':1},m.DEFAULT_POLICY,'pair')
   with self.assertRaises(ValueError):m.validate_admission(a,b,m.DEFAULT_POLICY,'final')
   v.write_text('changed')
   with self.assertRaises(ValueError):m.validate_admission(a,b,m.DEFAULT_POLICY,'pair')
 def test_explicit_cancellation_settles(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   c=Path(d)/'cancel';c.touch()
   t=m.supervise([sys.executable,'-c','import time; time.sleep(5)'],Path(d)/'cancelled',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.01),cancel_file=c)
   self.assertEqual(t['stop_reason'],'explicit_cancel');self.assertTrue(t['child_waited']);self.assertTrue(t['process_group_absent'])
 def test_large_terminal_tail_is_fully_drained(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
   script='import sys; sys.stderr.write((\'FMLSIRM_DIF_PROGRESS {"event":"optimizer_terminal"}\\n\')*2000)'
   t=m.supervise([sys.executable,'-c',script],Path(d)/'tail',dict(os.environ),dict(m.DEFAULT_POLICY,poll_seconds=0.1))
   self.assertTrue(m.successful_terminal(t),t);self.assertEqual(t['progress_records'],2000)
if __name__=='__main__':unittest.main(verbosity=2)
