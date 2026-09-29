"""Edge cases in the curation logic (src/curation.py).

Every test points the curation module's paths at a temporary directory, so running the
tests never touches the real data/ or skills/ folders.
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import curation  # noqa: E402


class TestCurationEdgeCases(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.test_dir, 'curation_log.json')
        self.memory_dir = os.path.join(self.test_dir, 'persistent_memory')
        self.skills_dir = os.path.join(self.test_dir, 'skills')
        os.makedirs(self.skills_dir)
        for name, value in (('CURATION_LOG', self.log_path),
                            ('PERSISTENT_MEMORY_DIR', self.memory_dir),
                            ('SKILLS_DIR', self.skills_dir),
                            ('SKILL_CREATION_DIR', self.skills_dir)):
            patcher = mock.patch.object(curation, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

        self.test_skill_data = {
            'path': os.path.join(self.skills_dir, 'test_skill.md'),
            'content': '# Test Skill\n',
            'metadata': {'name': 'Test Skill', 'description': 'A test skill for curation'},
        }
        with open(self.test_skill_data['path'], 'w') as f:
            f.write(self.test_skill_data['content'])

    def test_missing_metadata(self):
        results = curation.curate_skills([{'path': 'bad_skill.md', 'metadata': {}}])

        self.assertEqual(results[0]['action'], 'error')
        self.assertIn('Missing metadata', results[0]['error'])
        history = curation.load_curation_history()
        self.assertEqual(history[-1]['results'], results)

    def test_unexpected_error(self):
        # A skill that needs an update, with update_skill failing
        skill = dict(self.test_skill_data, metadata={'execution_count': 1, 'error_rate': 0.5})
        with mock.patch.object(curation, 'update_skill', side_effect=ValueError('Intentionally triggered error')):
            results = curation.curate_skills([skill])

        self.assertEqual(results[0]['action'], 'error')
        self.assertIn('ValueError', results[0]['error'])
        self.assertEqual(curation.load_curation_history()[-1]['results'], results)

    def test_update_is_applied(self):
        skill = dict(self.test_skill_data, metadata={'execution_count': 1, 'error_rate': 0.5})
        results = curation.curate_skills([skill])

        self.assertEqual(results[0]['action'], 'updated')
        with open(skill['path']) as f:
            self.assertIn('## Curated on', f.read())

    def test_persistent_memory_update(self):
        results = curation.curate_skills([self.test_skill_data])

        self.assertEqual(results[0]['action'], 'kept')
        self.assertEqual(curation.load_curation_history()[-1]['results'], results)
        with open(self.log_path) as f:
            self.assertEqual(json.load(f)[-1]['results'], results)
        with open(os.path.join(self.memory_dir, 'memory.json')) as f:
            self.assertEqual(json.load(f)['results'], results)

    def test_history_accumulates(self):
        curation.curate_skills([self.test_skill_data])
        curation.curate_skills([])
        self.assertEqual(len(curation.load_curation_history()), 2)

    def test_critical_error_handling(self):
        # A run that fails as a whole is recorded in the history instead of raising
        with mock.patch.object(curation, 'load_skills', side_effect=OSError('disk gone')):
            results = curation.curate_skills()

        self.assertEqual(results, [])
        self.assertEqual(curation.load_curation_history()[-1]['error'], 'disk gone')

    def test_no_skills_processed(self):
        results = curation.curate_skills([])

        self.assertEqual(results, [])
        self.assertEqual(curation.load_curation_history()[-1]['results'], [])

    def test_default_curates_the_skills_directory(self):
        results = curation.curate_skills()
        self.assertEqual([r['skill'] for r in results], ['test skill'])

    def test_curation_log_creation(self):
        self.assertFalse(os.path.exists(self.log_path))
        curation.curate_skills([self.test_skill_data])

        with open(self.log_path) as f:
            self.assertEqual(len(json.load(f)), 1)


if __name__ == '__main__':
    unittest.main()
