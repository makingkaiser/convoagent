import unittest
from app import app

class TestApp(unittest.TestCase):
    def test_execute_python_function(self):
        with app.test_client() as client:
            response = client.post('/api/execute-gpt-query/test', json={'input': 'Hello'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('output', response.json)

class TestUploadFile(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_upload_file_with_pdf(self):
        with open('test.pdf', 'rb') as f:
            response = self.app.post('/upload/test', data={'file': f})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'File uploaded and processed successfully', response.data)