# Generated by Django 3.0.6 on 2022-03-29 08:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('recommend', '0018_auto_20220329_1352'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mylist',
            name='movie',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recommend.Movie3'),
        ),
        migrations.AlterField(
            model_name='myrating',
            name='movie',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recommend.Movie3'),
        ),
    ]
