from django.test import TestCase
from main.models import MongoDBModel  # Replace with your actual import path

class MongoDBModelTestCase(TestCase):
    def setUp(self):
        # Set up any necessary data for the tests
        MongoDBModel.objects.create(field1='Test Field', field2=42)

    def test_model_fields(self):
        # Get an instance of the model from the database
        obj = MongoDBModel.objects.get(field1='Test Field')

        # Test the fields of the model
        self.assertEqual(obj.field1, 'Test Field')
        self.assertEqual(obj.field2, 42)

    # Add more test methods as needed
