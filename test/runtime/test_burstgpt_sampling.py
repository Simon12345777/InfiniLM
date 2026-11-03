"""Regression tests for length-distribution selection, not GPU execution."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / 'test_chunk_prefill'))
from burstgpt_workload import sample_burstgpt


class BurstGPTSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name) / 'data.csv'
        self.csv.write_text('Model,Request tokens,Response tokens\n'
                            'GPT-4,128,32\nGPT-4,4096,1000\n'
                            'GPT-4,9000,20\nGPT-4,bad,4\nGPT-4,3,0\n'
                            'ChatGPT,2048,40\n', encoding='utf-8')

    def args(self, **kw):
        values = dict(dataset_path=str(self.csv), num_prompts=2, warmup_requests=1,
            seed=0, use_full_dataset=False, max_request_tokens=8192,
            max_response_tokens=256, prefill_chunk_size=1024,
            burstgpt_model='GPT-4', burstgpt_output_policy='cap', allow_repeat_samples=False)
        values.update(kw)
        return SimpleNamespace(**values)

    def test_cap_retains_long_input_and_records_original_output(self):
        lengths, r = sample_burstgpt(self.args())
        self.assertEqual(set(lengths), {(128, 32), (4096, 256)})
        self.assertEqual(r['selected_output_capped'], 1)
        self.assertIn(1000, r['source_output_lens'])
        self.assertEqual(r['counts']['invalid_lengths'], 2)
        self.assertEqual(r['counts']['excluded_input_limit'], 1)

    def test_filter_preserves_both_lengths(self):
        lengths, r = sample_burstgpt(self.args(burstgpt_output_policy='filter', num_prompts=1))
        self.assertEqual(lengths, [(128, 32)])
        self.assertEqual(r['counts']['excluded_output_limit'], 1)
        self.assertEqual(r['selected_output_capped'], 0)

    def test_insufficient_rows_require_explicit_replacement(self):
        with self.assertRaisesRegex(ValueError, 'only 2 eligible'):
            sample_burstgpt(self.args(num_prompts=5))
        lengths, r = sample_burstgpt(self.args(num_prompts=5, allow_repeat_samples=True))
        self.assertEqual(len(lengths), 5)
        self.assertTrue(r['sampling_with_replacement'])
        self.assertLessEqual(r['selected_unique_rows'], 2)

    def test_deterministic_sampling_and_separate_warmup_report(self):
        a = self.args()
        self.assertEqual(sample_burstgpt(a), sample_burstgpt(a))
        _, measured = sample_burstgpt(a)
        _, warmup = sample_burstgpt(a, warmup=True)
        self.assertEqual(measured['selected_rows'], 2)
        self.assertEqual(warmup['selected_rows'], 1)
        self.assertNotEqual(measured['seed'], warmup['seed'])

    def test_full_dataset_bypasses_caps_explicitly(self):
        lengths, r = sample_burstgpt(self.args(use_full_dataset=True, num_prompts=3))
        self.assertIn((4096, 1000), lengths)
        self.assertIn((9000, 20), lengths)
        self.assertEqual(r['output_policy'], 'uncapped')
        self.assertEqual(r['selected_output_capped'], 0)

    def test_model_selection_and_zero_warmup(self):
        lengths, _ = sample_burstgpt(self.args(burstgpt_model='ChatGPT', num_prompts=1))
        self.assertEqual(lengths, [(2048, 40)])
        lengths, r = sample_burstgpt(self.args(warmup_requests=0), warmup=True)
        self.assertEqual(lengths, [])
        self.assertEqual(r['selected_unique_rows'], 0)

    def test_schema_and_all_invalid(self):
        self.csv.write_text('wrong,columns\n1,2\n')
        with self.assertRaisesRegex(ValueError, 'requires columns'):
            sample_burstgpt(self.args())
        self.csv.write_text('Model,Request tokens,Response tokens\nGPT-4,0,2\n')
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            sample_burstgpt(self.args())


if __name__ == '__main__':
    unittest.main()
