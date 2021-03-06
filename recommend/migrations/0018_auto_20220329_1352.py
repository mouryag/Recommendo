# Generated by Django 3.0.6 on 2022-03-29 08:22

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('recommend', '0017_auto_20220326_1448'),
    ]

    operations = [
        migrations.CreateModel(
            name='Movie4',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('adult', models.BooleanField()),
                ('belongs_to_collection', models.TextField()),
                ('budget', models.FloatField()),
                ('genres', models.TextField()),
                ('homepage', models.CharField(max_length=400)),
                ('imp', models.IntegerField()),
                ('imdb_id', models.CharField(max_length=200)),
                ('original_language', models.CharField(max_length=200)),
                ('original_title', models.CharField(max_length=200)),
                ('overview', models.TextField()),
                ('popularity', models.FloatField()),
                ('poster_path', models.CharField(max_length=200)),
                ('production_companies', models.TextField()),
                ('production_countries', models.TextField()),
                ('release_date', models.CharField(max_length=200)),
                ('revenue', models.FloatField()),
                ('runtime', models.FloatField()),
                ('spoken_languages', models.TextField()),
                ('status', models.CharField(max_length=200)),
                ('tagline', models.CharField(max_length=400)),
                ('title', models.CharField(max_length=200)),
                ('video', models.TextField()),
                ('vote_average', models.FloatField()),
                ('vote_count', models.IntegerField()),
                ('imdb_url', models.CharField(max_length=200)),
                ('year', models.IntegerField()),
                ('img_url', models.TextField()),
            ],
        ),
        migrations.AlterField(
            model_name='myrating',
            name='rating',
            field=models.IntegerField(default=0, validators=[django.core.validators.MaxValueValidator(10), django.core.validators.MinValueValidator(0)]),
        ),
        migrations.AlterField(
            model_name='mylist',
            name='movie',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recommend.Movie4'),
        ),
        migrations.AlterField(
            model_name='myrating',
            name='movie',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recommend.Movie4'),
        ),
        migrations.AlterUniqueTogether(
            name='myrating',
            unique_together={('user', 'movie')},
        ),
        migrations.AlterIndexTogether(
            name='myrating',
            index_together={('user', 'movie')},
        ),
    ]
