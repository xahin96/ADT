# admin.py
from django.contrib import admin
from .models import MongoDBModel, PostgreSQLModel

class MongoDBModelAdmin(admin.ModelAdmin):
    list_display = ['field1', 'field2']  # Customize as needed

class PostgreSQLModelAdmin(admin.ModelAdmin):
    list_display = ['postgres_field']  # Customize as needed

admin.site.register(MongoDBModel, MongoDBModelAdmin)
admin.site.register(PostgreSQLModel, PostgreSQLModelAdmin)
