from django.db import models

from djongo import models as djongo_models

class MongoDBModel(djongo_models.Model):
    field1 = models.CharField(max_length=255)
    field2 = models.IntegerField()

class PostgreSQLModel(models.Model):
    postgres_field = models.CharField(max_length=255)
