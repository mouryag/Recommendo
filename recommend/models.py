from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User


# Create your models here.

class Movie(models.Model):
    title = models.CharField(max_length=200)
    genre = models.CharField(max_length=100)
    movie_logo = models.FileField()

    def __str__(self):
        return self.title

class Movie3(models.Model):
    adult = models.BooleanField()
    belongs_to_collection= models.TextField()
    budget = models.FloatField()
    genres = models.TextField()
    homepage= models.CharField(max_length=400)
    #imp=models.IntegerField()
    imdb_id = models.CharField(max_length=200)
    original_language=models.CharField(max_length=200)
    original_title=models.CharField(max_length=200)
    overview= models.TextField()
    popularity= models.FloatField()
    poster_path = models.CharField(max_length=200)
    production_companies = models.TextField()
    production_countries = models.TextField()
    release_date= models.CharField(max_length=200)
    revenue = models.FloatField()
    runtime = models.FloatField()
    spoken_languages = models.TextField()
    status = models.CharField(max_length=200)
    tagline=models.CharField(max_length=400)
    title=models.CharField(max_length=200)
    video=models.TextField()
    vote_average = models.FloatField()
    vote_count = models.IntegerField()
    imdb_url=models.CharField(max_length=200)
    year = models.IntegerField()
    img_url= models.TextField()


    def __str__(self):
        return self.title

class Movie4(models.Model):
    adult = models.BooleanField()
    belongs_to_collection= models.TextField()
    budget = models.FloatField()
    genres = models.TextField()
    homepage= models.CharField(max_length=400)
    imp=models.IntegerField()
    imdb_id = models.CharField(max_length=200)
    original_language=models.CharField(max_length=200)
    original_title=models.CharField(max_length=200)
    overview= models.TextField()
    popularity= models.FloatField()
    poster_path = models.CharField(max_length=200)
    production_companies = models.TextField()
    production_countries = models.TextField()
    release_date= models.CharField(max_length=200)
    revenue = models.FloatField()
    runtime = models.FloatField()
    spoken_languages = models.TextField()
    status = models.CharField(max_length=200)
    tagline=models.CharField(max_length=400)
    title=models.CharField(max_length=200)
    video=models.TextField()
    vote_average = models.FloatField()
    vote_count = models.IntegerField()
    imdb_url=models.CharField(max_length=200)
    year = models.IntegerField()
    img_url= models.TextField()


    def __str__(self):
        return self.title



class Myrating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie4, on_delete=models.CASCADE)
    rating = models.IntegerField(default=0, validators=[MaxValueValidator(10), MinValueValidator(0)])

    class Meta:
        unique_together = (("user", "movie"),)
        index_together = (("user", "movie"),)

    def __str__(self):
        return str(self.user)+"_"+str(self.movie)+"_"+str(self.rating)

class MyList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie4, on_delete=models.CASCADE)
    watch = models.BooleanField(default=False)
