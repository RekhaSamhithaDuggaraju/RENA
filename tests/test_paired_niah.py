import unittest
from core.llm.paired_niah import compare_niah_to_reference


def payload(retrieved):
    rows=[]
    for fid,split,mode,val in [('f1','forget','absent',retrieved[0]),('f1','forget','present',retrieved[1]),('r1','retain','present',retrieved[2])]:
        rows.append({'case_id':f'{fid}-{mode}','fact_id':fid,'target_split':split,'context_mode':mode,'insertion_position':'middle','prompt_style':'direct','context_length_chars':100,'retrieved':val})
    return {'cases':rows}

class PairedNiahTests(unittest.TestCase):
    def test_matched_reference_deltas(self):
        out=compare_niah_to_reference(payload([False,True,True]),payload([True,True,True]),bootstrap_iterations=100)
        self.assertEqual(out['matched_case_count'],3)
        self.assertEqual(out['forget_absent_disclosure']['difference'],-1.0)
        self.assertEqual(out['retain_present_retrieval']['difference'],0.0)
    def test_rejects_unmatched_metadata(self):
        a=payload([False,True,True]); b=payload([True,True,True]); b['cases'][0]['context_length_chars']=999
        with self.assertRaises(ValueError): compare_niah_to_reference(a,b,bootstrap_iterations=100)
    def test_rejects_no_shared_cases(self):
        a=payload([False,True,True]); b=payload([True,True,True]); [row.update({'case_id': 'other-' + row['case_id']}) for row in b['cases']]
        with self.assertRaises(ValueError): compare_niah_to_reference(a,b,bootstrap_iterations=100)

if __name__=='__main__': unittest.main()
