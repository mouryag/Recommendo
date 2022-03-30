from django.contrib import admin
from .models import Movie, Myrating, MyList, Movie3,Movie4

# Register your models here.
admin.site.register(Movie)
admin.site.register(Movie3)
admin.site.register(Movie4)
admin.site.register(Myrating)
admin.site.register(MyList)