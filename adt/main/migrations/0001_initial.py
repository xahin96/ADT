# Generated by Django 4.1 on 2024-01-18 02:36

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MongoDBModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('field1', models.CharField(max_length=255)),
                ('field2', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='PostgreSQLModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('postgres_field', models.CharField(max_length=255)),
            ],
        ),
    ]
