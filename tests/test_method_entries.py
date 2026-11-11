import unittest
from methods.launch import ROOT, build_command, load_spec

class MethodEntries(unittest.TestCase):
    def test_forward_arguments_without_interpretation(self):
        args=['--config','a file.json','train','--model','t5_base']
        self.assertEqual(build_command('baseline','run',args)[1:],['-m','ppnl.cli',*args])

    def test_catalog_paths_exist(self):
        for name in ('baseline','gvr','harness','lora','rl','prefix','policy'):
            spec=load_spec(name)
            for path in spec['configs']+spec['modules']:
                self.assertTrue((ROOT/path).exists(),path)
            for command in spec['commands'].values():
                if command[0]!='-m': self.assertTrue((ROOT/command[0]).is_file())

    def test_reject_unknown_routes(self):
        with self.assertRaises(ValueError): load_spec('../baseline')
        with self.assertRaises(ValueError): build_command('rl','unknown',[])
