import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def load(name,filename):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
pub=load('attention_publish','publish_repo.py');src=load('attention_sources','check_sources.py')
class BootstrapTests(unittest.TestCase):
    def test_archived_source_hashes(self):self.assertEqual(src.verify(),[])
    def test_origin_strict(self):
        self.assertTrue(pub.origin_matches('https://github.com/wlvh/attention-lab.git','wlvh/attention-lab'))
        self.assertTrue(pub.origin_matches('git@github.com:wlvh/attention-lab.git','wlvh/attention-lab'))
        self.assertFalse(pub.origin_matches('https://evil.example/wlvh/attention-lab','wlvh/attention-lab'))
        self.assertFalse(pub.origin_matches('https://github.com/wlvh/other.git','wlvh/attention-lab'))
    def test_name_validation(self):
        for owner,name in [('wlvh/other','attention-lab'),('wlvh','../x'),('wlvh','x;rm')]:
            with self.assertRaises(pub.PublishError):pub.safe_name(owner,name)
    def test_dry_run_never_publishes(self):
        with patch.object(pub,'publish',side_effect=AssertionError('must not execute')),patch('builtins.print'):
            self.assertEqual(pub.main([]),0)
    def test_missing_cli_fails_without_commands(self):
        with patch.object(pub.shutil,'which',return_value=None),patch.object(pub,'command',side_effect=AssertionError('no commands')):
            with self.assertRaises(pub.PublishError):pub.publish('wlvh','attention-lab')
    def test_wrong_account_stops_before_git_or_creation(self):
        def run(args,**kw):
            if args[:3]==['gh','auth','status']:return subprocess.CompletedProcess(args,0,'','')
            if args[:3]==['gh','api','user']:return subprocess.CompletedProcess(args,0,'another-user\n','')
            raise AssertionError(args)
        with patch.object(pub.shutil,'which',return_value='/mock/bin'),patch.object(pub,'command',side_effect=run):
            with self.assertRaises(pub.PublishError):pub.publish('wlvh','attention-lab')
    def test_public_remote_rejected(self):
        d={'nameWithOwner':'wlvh/attention-lab','isPrivate':False,'url':'https://github.com/wlvh/attention-lab'}
        with patch.object(pub,'command',return_value=subprocess.CompletedProcess([],0,json.dumps(d),'')):
            with self.assertRaises(pub.PublishError):pub.remote_info('wlvh/attention-lab')
    def test_manifest_rejects_wrong_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(pub.PublishError):pub.verify_bundle(Path(d))
if __name__=='__main__':unittest.main()
