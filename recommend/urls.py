from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r"^avg/(?P<voteavg>\d+(\.\d+)?)/(?P<yr>\d+)/$",views.voteavg,name='avg'),
    url(r"^gvy/(?P<genre>\w+)/(?P<v>\d+(\.\d+)?)/(?P<y>\d+)/$",views.gvy,name='gvy'),
    url(r"^gy/(?P<genre>\w+)/(?P<y>\d+)/$",views.gy,name='gy'),
    path('franchise/<str:col>',views.franchise,name='franchise'),
    path('prod/<str:house>',views.prod,name='prod'),
    path('actor/<str:actr>',views.actor,name='actor'),
    path('actor2/<str:actr2>',views.actor2,name='actor2'),
    path('director/<str:dir>',views.director,name='director'),
    path('genre/<str:genre>',views.genres,name='genres'),
    path('signup/', views.signUp, name='signup'),
    path('login/', views.Login, name='login'),
    path('logout/', views.Logout, name='logout'),
    path('<int:movie_id>/', views.detail, name='detail'),
    path('watch/', views.watch, name='watch'),
    path('recommend/', views.recommend, name='recommend'),
]