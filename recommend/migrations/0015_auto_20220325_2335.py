# Generated by Django 3.0.6 on 2022-03-25 18:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recommend', '0014_movie2'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie2',
            name='id',
            field=models.IntegerField(primary_key=True, serialize=False),
        ),
    ]
