import sys
sys.path.append('src')
import unittest
import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.curation import update_skill
import json
from src.curation import curate_skills, PERSISTENT_MEMORY_DIR, CURATION_LOG, load_curation_history
from src.tool_parser import parse_tool_call
import tempfile
import shutil

class TestCurationEdgeCases(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_curation_log = 'data/curation_log.json'
        self.original_persistent_dir = 'data/persistent_memory'
        
        # Set test paths
        CURATION_LOG = os.path.join(self.test_dir, 'curation_log.json')
        PERSISTENT_MEMORY_DIR = os.path.join(self.test_dir, 'persistent_memory/')
        
        # Create test skill data
        self.test_skill_data = {'path': 'test_skill.md', 'metadata': {'name': 'Test Skill', 'description': 'A test skill for curation'}}

    def tearDown(self):
        # Restore original paths
        CURATION_LOG = self.original_curation_log
        PERSISTENT_MEMORY_DIR = self.original_persistent_dir
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_missing_metadata(self):
        # Test handling of missing metadata in skill data
        bad_skill = {
            'path': 'bad_skill.md',
            'metadata': {}
        }
        
        # Run curation with bad skill data
        results = curate_skills([bad_skill])
        
        # Check if error was properly recorded
        self.assertTrue(any(result['action'] == 'error' for result in results))
        self.assertTrue(any('Missing metadata' in result['error'] for result in results))
        
        # Verify persistent memory was updated
        history = load_curation_history()
        self.assertTrue(len(history) > 0)
        self.assertTrue('error' in history[-1])

    def test_unexpected_error(self):
        # Patch update_skill to raise error
        original_update_skill = update_skill
        def faulty_update(skill):
            raise ValueError("Intentionally triggered error")
        curate_skills.update_skill = faulty_update

        try:
            results = curate_skills([self.test_skill_data])  # Ensure list is passed

            # Check if error was properly recorded
            self.assertTrue(any(result['action'] == 'error' for result in results))
            self.assertTrue(any('ValueError' in result['error'] for result in results))

            # Verify persistent memory was updated
            history = load_curation_history()
            self.assertTrue(len(history) > 0)
            self.assertTrue('error' in history[-1])
        finally:
            # Restore original function
            curate_skills.update_skill = original_update_skill
        # Test handling of unexpected errors during curation
        def faulty_update(skill):
            raise ValueError("Intentionally triggered error")
        
        # Patch update_skill function to raise error
        original_update_skill = update_skill
        curate_skills.update_skill = faulty_update
        
        try:
            results = curate_skills(self.test_skill_data)
            
            # Check if error was properly recorded
            self.assertTrue(any(result['action'] == 'error' for result in results))
            self.assertTrue(any('ValueError' in result['error'] for result in results))
            
            # Verify persistent memory was updated
            history = load_curation_history()
            self.assertTrue(len(history) > 0)
            self.assertTrue('error' in history[-1])
        finally:
            # Restore original function
            curate_skills.update_skill = original_update_skill

    def test_persistent_memory_update(self):
        # Test that persistent memory is properly updated
        results = curate_skills(self.test_skill_data)
        
        # Verify history was updated
        history = load_curation_history()
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[-1]['results'], results)
        
        # Verify curation log was updated
        with open(CURATION_LOG, 'r') as f:
            log_data = json.load(f)
        self.assertEqual(log_data[-1]['results'], results)

    def test_critical_error_handling(self):
        # Test handling of critical errors during curation
        # Create persistent memory directory first
        os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)
        # Remove persistent memory directory to trigger error
        os.rmdir(PERSISTENT_MEMORY_DIR)

        try:
            curate_skills([self.test_skill_data])
        except Exception as e:
            # Ensure error is caught and logged
            self.assertTrue('error' in load_curation_history()[-1])

            # Verify error was recorded in persistent memory
            history = load_curation_history()
            self.assertTrue(len(history) > 0)
            self.assertTrue('error' in history[-1])
        finally:
            # Clean up
            os.makedirs(PERSISTENT_MEMORY_DIR)

    def test_no_skills_processed(self):
        # Test case where no skills are processed
        results = curate_skills([])
        
        # Verify results are empty
        self.assertEqual(len(results), 0)
        
        # Verify persistent memory was updated
        history = load_curation_history()
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[-1]['results'], [])

    def test_curation_log_creation(self):
        # Test that curation log is created if it doesn't exist
        # Remove existing log file if it exists
        if os.path.exists(CURATION_LOG):
            os.remove(CURATION_LOG)

        # Run curation with a list of skills
        curate_skills([self.test_skill_data])
        
        # Verify log file was created
        self.assertTrue(os.path.exists(CURATION_LOG))
        
        # Verify log content
        with open(CURATION_LOG, 'r') as f:
            log_data = json.load(f)
        self.assertTrue(len(log_data) > 0)

if __name__ == '__main__':
    unittest.main()