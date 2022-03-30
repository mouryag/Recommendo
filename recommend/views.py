from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .forms import *
from django.http import Http404
from django.template import loader
from .models import Movie, Myrating, MyList,Movie3,Movie4
from django.db.models import Q
from django.contrib import messages
from django.http import HttpResponseRedirect,StreamingHttpResponse,HttpResponse
from django.db.models import Case, When
import pandas as pd
import requests
import numpy as np
import ast
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from nltk.stem.snowball import SnowballStemmer
import time

# Create your views here.
def stod(s):
    try:
        return ast.literal_eval(s)
    except:
        return None
u=[]
md = pd.read_csv('meta_credits5.csv')
#md = pd.DataFrame(list(Movie4.objects.all().values()))
print("+++++++++++++++++++++++++++++++++++++++++++++++++STARTING++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#links_small = pd.read_csv('links_small.csv')
#credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
#print(credits.head())
#print(keywords.head())
keywords['imp'] = keywords['id'].astype('int')
#credits['imp'] = credits['id'].astype('int')
md['imp'] = md['imp'].astype('int')
#links_small = links_small[links_small['imdbId'].notnull()]['imdbId'].astype('int')
#md=md[['belongs_to_collection','genres','imp','imdb_id','production_companies','title']]
#md = md.merge(credits, on='imp')
#md.to_csv("meta_credits5.csv")
md = md.merge(keywords, on='imp')
md['imdb_id'] = md['imdb_id'].apply(lambda x: x[3:]).astype('int')
md['genres'] = md['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2 = md
#print(smd2.shape)
smd2['cast'] = smd2['cast'].apply(ast.literal_eval)
#print(smd2['cast'][2])
smd2['crew'] = smd2['crew'].apply(ast.literal_eval)
smd2['keywords'] = smd2['keywords'].apply(ast.literal_eval)
smd2['cast_size'] = smd2['cast'].apply(lambda x: len(x))
smd2['crew_size'] = smd2['crew'].apply(lambda x: len(x))
#print(smd2['cast'])
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
smd2['director'] = smd2['crew'].apply(get_director)
dirs = smd2['director'].value_counts()
dirs = dirs[dirs>6]
d = list(dirs.index)
#print(smd2[smd2['director']=='Steven Spielberg'])
smd2['director2'] = smd2['director']


def ext(x):
    for i in x:
        return i['name']
def ext2(x):
    try:
        for i in x:
            if i['gender']==1:
                return i['name']
    except:
        return None
smd2['cast2'] = smd2['cast'].apply(ext)
smd2['cast3'] = smd2['cast'].apply(ext2)
actors = smd2['cast2'].value_counts()
actors = actors[actors>9]
act = list(actors.index)

actors2 = smd2['cast3'].value_counts()
actors2 = actors2[actors2>9]
act2 = list(actors2.index)
#print(act2)
#print(len(act2))
#print(smd2['cast2'])
smd2['cast'] = smd2['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2['cast'] = smd2['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd2['keywords'] = smd2['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd2['cast'] = smd2['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd2['director'] = smd2['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd2['director'] = smd2['director'].apply(lambda x: [x,x, x])
s = smd2.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]
#stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
smd2['keywords'] = smd2['keywords'].apply(filter_keywords)
#smd2['keywords'] = smd2['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd2['keywords'] = smd2['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd2['soup'] = smd2['keywords'] + smd2['cast'] + smd2['director'] + smd2['genres']
smd2['soup'] = smd2['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

smd2 = smd2.reset_index()
titles = smd2['title']
indices = pd.Series(smd2.index, index=smd2['title'])

def get_recommendations2(title):
    idx = indices[title]
    #print(idx)
    if isinstance(idx, np.int64):
        pass
    else:
        idx=1536
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']

# smd = md
# #smd = md[md['imdb_id'].isin(links_small)]
# #print(smd.shape)
#
# smd['tagline'] = smd['tagline'].fillna('')
# smd['description'] = smd['overview'] + smd['tagline']
# smd['description'] = smd['description'].fillna('')
#
# tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(smd['description'])
#
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# smd = smd.reset_index()
# titles = smd['title']
# indices = pd.Series(smd.index, index=smd['title'])
#
# reclist = list(titles.values)
# def get_recommendations(title):
#     idx = indices[title]
#     print(idx)
#     if isinstance(idx, np.int64):
#         pass
#     else:
#         idx=1536
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:31]
#     movie_indices = [i[0] for i in sim_scores]
#     return titles.iloc[movie_indices]

#print(get_recommendations('Jumanji').head(10).values)
#print(tfidf_matrix.shape)
#print(df['belongs_to_collection'])
df_fran = md[md['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(stod)
#print(df_fran['belongs_to_collection'][20:35])
df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection']=df_fran['belongs_to_collection'].apply(lambda x: x.get('name',"none") if isinstance(x, dict) else np.nan)
df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]
coll = list(df_fran['belongs_to_collection'].value_counts().index)[:30]
df_pc = md[md['production_companies'].notnull()]
df_pc['production_companies'] = df_pc['production_companies'].fillna('[]')
df_pc['production_companies'] = df_pc['production_companies'].apply(stod)
df_pc = df_pc[df_pc['production_companies'].notnull()]
df_pc['production_companies'] = df_pc['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
pc ={}
for i in df_pc['production_companies'].values:
    for c in i:
        if c not in pc:
            pc[c]=1
        else:
            pc[c]+=1
pcl = {}
for k in pc.keys():
    if pc[k]>=23:
        pcl[k]=pc[k]
#print(pcl)
comps=list(pcl.keys())
df_pc['production_companies'] = df_pc['production_companies'].apply(str)
#smd2['cast2'] = smd2['cast2'].apply(str)
#print(list(df_pc[df_pc['production_companies'].str.contains(pat = 'Pixar Animation Studios')]['id'].values))


def actor(request,actr):
    txt = actr + " Movies"
    mvids = list(smd2[smd2['cast2']==actr]['title'].index)
    #print(mvids)
    #print(mvids)
    mvs = Movie4.objects.filter(pk__in=mvids)
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    #print(mvs)
    return render(request, 'recommend/list.html', {'coll':coll,'movies':mvs,'no':True,'genres':genres,'txt':txt,'comps':comps,'dirs':d,'act':act,'act2':act2})


def actor2(request,actr2):
    txt = actr2 + " Movies"
    mvids = list(smd2[smd2['cast3']==actr2]['title'].index)
    #print(mvids)
    #print(mvids)
    mvs = Movie4.objects.filter(pk__in=mvids)
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    #print(mvs)
    return render(request, 'recommend/list.html', {'coll':coll,'movies':mvs,'no':True,'genres':genres,'txt':txt,'comps':comps,'dirs':d,'act':act,'act2':act2})

def director(request,dir):
    txt = dir + " Movies"
    mvids = list(smd2[smd2['director2']==dir]['title'].index)
    #print(mvids)
    mvs = Movie4.objects.filter(pk__in=mvids)
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    #print(mvs)
    return render(request, 'recommend/list.html', {'coll':coll,'movies':mvs,'no':True,'genres':genres,'txt':txt,'comps':comps,'dirs':d,'act':act,'act2':act2})

def prod(request,house):
    txt = house + " Movies"
    mvids = list(df_pc[df_pc['production_companies'].str.contains(house)]['title'].index)
    #print(mvids)
    mvs = Movie4.objects.filter(pk__in=mvids)
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    #print(mvs)
    return render(request, 'recommend/list.html', {'coll':coll,'movies':mvs,'no':True,'genres':genres,'txt':txt,'comps':comps,'dirs':d,'act':act,'act2':act2})

def franchise(request,col):
    txt = col + " Movies"
    mvids = list(df_fran[df_fran['belongs_to_collection']==col]['title'].index)
    #print(mvids)
    mvs = Movie4.objects.filter(pk__in=mvids)
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    #print(mvs)
    return render(request, 'recommend/list.html', {'coll':coll,'movies':mvs,'no':True,'genres':genres,'txt':txt,'comps':comps,'dirs':d,'act':act,'act2':act2})
def stod(s):
    try:
        return ast.literal_eval(s)
    except:
        return None

def hello():
    # yield 'Hello,'
    # yield 'there!'
    imdb_ids = ['tt0114709','tt0113497','tt0113228','tt0114885','tt0113041','tt0113277','tt0114319','tt0112302','tt0114576','tt0113189','tt0112346',
     'tt0112896','tt0112453','tt0113987']
    for id in imdb_ids:
        URL = "https://www.imdb.com/title/"+id+"/?ref_=fn_al_tt_1"
        r = requests.get(URL)
        soup = BeautifulSoup(r.content, 'html5lib')
        mv = soup.find_all("img", class_="ipc-image")[0]['src']
        yield mv
    # for x in range(1,9):
    #     yield str(x) # Returns a chunk of the response to the browser
    #     time.sleep(1)

def my_view(request):
    # NOTE: No Content-Length header!
    return StreamingHttpResponse(hello())
def render_gen(template, lst):
    for l in lst:
        yield template.render({'l':l})

# def index(request):
#     movies = Movie.objects.all()
#     query = request.GET.get('q')
#     URL = "https://www.imdb.com/title/tt0113228/?ref_=fn_al_tt_1"
#     r = requests.get(URL)
#     soup = BeautifulSoup(r.content, 'html5lib')
#     mv = soup.find_all("img", class_="ipc-image")[0]['src']
#     lst = hello()
#     template = loader.get_template('recommend/list.html')
#
#     # if query:
#     #     movies = Movie.objects.filter(Q(title__icontains=query)).distinct()
#     #     return render(request, 'recommend/list.html', {'movies': movies})
#
#     # return render(request, 'recommend/list.html', {'movies': movies,'mv':mv})
#     return StreamingHttpResponse(render_gen(template,lst))
def gy(request,genre,y):
    txt = genre+" Movies in year "+str(y)
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    mvs = Movie4.objects.filter(genres__icontains = genre,year=y).order_by("-year").order_by("-vote_average")
    no = True
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    return render(request, 'recommend/list.html', {'movies': mvs,'genres':genres,'no':no,'txt':txt,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

def gvy(request,genre,v,y):
    txt = genre+" Movies with rating "+str(v)+" and more in year "+str(y)
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    mvs = Movie4.objects.filter(genres__icontains = genre,vote_average__gte=v,year=y).order_by("-year").order_by("-vote_average")
    no = True
    query = request.GET.get('q')

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    if y == 1 or y == "1":
        txt = " All "+genre+" Movies with rating "+str(v)+" and more"
        mvs = Movie4.objects.filter(genres__icontains = genre,vote_average__gte=v).order_by("-year").order_by("-vote_average")
        query = request.GET.get('q')
        if query:
            movies = mvs.filter(Q(title__icontains=query)).distinct()
            return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

        return render(request, 'recommend/list.html', {'movies': mvs,'genres':genres,'no':no,'txt':txt,'coll':coll,'comps':comps,'dirs':d,'act':act,'act2':act2})
    return render(request, 'recommend/list.html', {'movies': mvs,'genres':genres,'no':no,'txt':txt,'coll':coll,'comps':comps,'dirs':d,'act':act,'act2':act2})

def genres(request,genre):
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    mvs = Movie4.objects.filter(genres__icontains = genre).order_by("-year")
    txt = genre + " Movies"
    no = False
    query = request.GET.get('q')
    p = Paginator(mvs, 100)
    page_number = request.GET.get('page')
    try:
        page_obj = p.get_page(page_number)  # returns the desired page object
    except PageNotAnInteger:
        # if page_number is not an integer then assign the first page
        page_obj = p.page(1)
    except EmptyPage:
        # if page is empty then return last page
        page_obj = p.page(p.num_pages)

    if query:
        movies = mvs.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    return render(request, 'recommend/list.html', {'page_obj': page_obj,'txt':txt,'genres':genres,'no':no})


def voteavg(request,voteavg,yr):
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    #mv = Movie4.objects.get(pk=mv_id)
    if yr == 1 or yr == "1":
        txt ="Movies with rating "+str(voteavg)+" and more"
        movies = Movie4.objects.filter(vote_average__gte=voteavg).order_by('-year').order_by("-vote_average")
        query = request.GET.get('q')

        p = Paginator(movies, 100)
        page_number = request.GET.get('page')
        try:
            page_obj = p.get_page(page_number)  # returns the desired page object
        except PageNotAnInteger:
            # if page_number is not an integer then assign the first page
            page_obj = p.page(1)
        except EmptyPage:
            # if page is empty then return last page
            page_obj = p.page(p.num_pages)

        if query:
            movies = movies.filter(Q(title__icontains=query)).distinct()
            return render(request, 'recommend/list.html', {'movies':movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

        return render(request, 'recommend/list.html', {'page_obj':page_obj,'txt':txt,'no':True,'genres':genres,'coll':coll,'comps':comps,'dirs':d,'act':act,'act2':act2})
    else:
        txt ="Movies with rating "+str(voteavg)+" and more in year "+str(yr)
        movies = Movie4.objects.filter(vote_average__gte=voteavg,year=yr).order_by('-year').order_by("-vote_average")
        query = request.GET.get('q')

        if query:
            movies = movies.filter(Q(title__icontains=query)).distinct()
            return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

        #print(movies)
        return render(request, 'recommend/list.html', {'movies': movies,'txt':txt,'no':True,'genres':genres,'coll':coll,'comps':comps,'dirs':d,'act':act,'act2':act2})



def index(request):
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    #mv = Movie4.objects.get(pk=mv_id)
    movies = Movie4.objects.all().order_by("-popularity")
    query = request.GET.get('q')

    p = Paginator(movies, 100)  # creating a paginator object
    # getting the desired page number from url
    page_number = request.GET.get('page')
    try:
        page_obj = p.get_page(page_number)  # returns the desired page object
    except PageNotAnInteger:
        # if page_number is not an integer then assign the first page
        page_obj = p.page(1)
    except EmptyPage:
        # if page is empty then return last page
        page_obj = p.page(p.num_pages)
    if query:
        movies = Movie4.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/list.html', {'movies': movies,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})

    return render(request, 'recommend/list.html', {'page_obj': page_obj,'genres':genres,'no':True,'coll':coll,'genres':genres,'comps':comps,'dirs':d,'act':act,'act2':act2})



# Show details of the movie
def detail(request, movie_id):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404
    movies = get_object_or_404(Movie4, id=movie_id)
    movie = Movie4.objects.get(id=movie_id)

    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime','Documentary', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation','Family']
    #mv = Movie4.objects.get(pk=mv_id)
    g = []
    for i in genres:
        if movie.genres != None and i in movie.genres:
            g.append(i)
    
    temp = list(MyList.objects.all().values().filter(movie_id=movie_id,user=request.user))
    if temp:
        update = temp[0]['watch']
    else:
        update = False
    if request.method == "POST":

        # For my list
        if 'watch' in request.POST:
            watch_flag = request.POST['watch']
            if watch_flag == 'on':
                update = True
            else:
                update = False
            if MyList.objects.all().values().filter(movie_id=movie_id,user=request.user):
                MyList.objects.all().values().filter(movie_id=movie_id,user=request.user).update(watch=update)
            else:
                q=MyList(user=request.user,movie=movie,watch=update)
                q.save()
            if update:
                messages.success(request, "Movie added to your list!")
            else:
                messages.success(request, "Movie removed from your list!")

            
        # For rating
        else:
            rate = request.POST['rating']
            if Movie4.objects.get(pk = movie_id):
                mve = Movie4.objects.get(pk = movie_id)
                #print(mve)
                #print(mve.vote_count,mve.vote_average)
                k = mve.vote_count
                k2 = mve.vote_average
                if request.user not in u:
                    mve.vote_count=k+1
                    u.append(request.user)
                #print(u)
                mve.vote_average = round((k2*k+int(rate))/(k+1),2)
                mve.save()
                #print(mve.vote_count,mve.vote_average)
            if Myrating.objects.all().values().filter(movie_id=movie_id,user=request.user):
                Myrating.objects.all().values().filter(movie_id=movie_id,user=request.user).update(rating=rate)
            else:
                q=Myrating(user=request.user,movie=movie,rating=rate)
                q.save()

            messages.success(request, "Rating has been submitted!")

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    out = list(Myrating.objects.filter(user=request.user.id).values())

    # To display ratings in the movie detail page
    movie_rating = 0
    rate_flag = False
    for each in out:
        if each['movie_id'] == movie_id:
            movie_rating = each['rating']
            rate_flag = True
            break
    recs =[]
    recs2 =[]
    recmvs= []
    recmvs2=[]
    # recs = list(get_recommendations(movie.title).head().values)
    # for i in recs:
    #     recmvs.append(Movie4.objects.get(title=i))
    recs2 = list(get_recommendations2(movie.title).head(10).values)
    for rec in recs2:
        recmvs2.append(Movie4.objects.get(title=rec))

    context = {'movies': movies,'movie_rating':movie_rating,'recmvs2':recmvs2,'rate_flag':rate_flag,'update':update,'gens':g}
    return render(request, 'recommend/detail.html', context)


# MyList functionality
def watch(request):

    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

    movies = Movie4.objects.filter(mylist__watch=True,mylist__user=request.user)
    query = request.GET.get('q')

    if query:
        movies = Movie4.objects.filter(Q(title__icontains=query)).distinct()
        return render(request, 'recommend/watch.html', {'movies': movies})

    return render(request, 'recommend/watch.html', {'movies': movies})


# To get similar movies based on user rating
def get_similar(movie_name,rating,corrMatrix):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

# Recommendation Algorithm
def recommend(request):

    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404


    movie_rating=pd.DataFrame(list(Myrating.objects.all().values()))
    #print(movie_rating)

    new_user=movie_rating.user_id.unique().shape[0]
    current_user_id= request.user.id
	# if new user not rated any movie
    # if current_user_id>new_user:
    #     #r = random.randint(1,1500)
    #     movie=Movie4.objects.get(id=58)
    #     q=Myrating(user=request.user,movie=movie,rating=0)
    #     q.save()


    userRatings = movie_rating.pivot_table(index=['user_id'],columns=['movie_id'],values='rating')
    #print(userRatings)
    userRatings = userRatings.fillna(0,axis=1)
    corrMatrix = userRatings.corr(method='pearson')

    user = pd.DataFrame(list(Myrating.objects.filter(user=request.user).values()))
    #print(user.shape)
    if user.shape[0]!=0:
        user = user.drop(['user_id','id'],axis=1)
        user_filtered = [tuple(x) for x in user.values]
        movie_id_watched = [each[0] for each in user_filtered]
        #print(movie_id_watched)

        similar_movies = pd.DataFrame()
        for movie,rating in user_filtered:
            similar_movies = similar_movies.append(get_similar(movie,rating,corrMatrix),ignore_index = True)

        movies_id = list(similar_movies.sum().sort_values(ascending=False).index)
        movies_id_recommend = [each for each in movies_id if each not in movie_id_watched]
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(movies_id_recommend)])
        movie_list=list(Movie4.objects.filter(id__in = movies_id_recommend).order_by(preserved)[:user.shape[0]])

        mvrated = list(Myrating.objects.filter(user=request.user).values())
        #pdmvrated = pd.DataFrame(list(Myrating.objects.filter(user=request.user).values()))
        #print(pdmvrated)
        recmvs2 = []
        reco={}
        for i in mvrated:
            mv = Movie4.objects.get(pk=i['movie_id'])
            #print(mv.title)
            recs2 = list(get_recommendations2(mv.title).head(10).values)
            for rec in recs2:
                rmv = Movie4.objects.filter(title=rec).first()
                v = rmv.vote_count
                R = rmv.vote_average
                m = 300
                C = 6.5
                reco[rmv.id] = i['rating']*((v/(v+m) * R) + (m/(m+v) * C))
        #print(reco)
        # for rec in recmvs2:
        #     print(type(pdmvrated[pdmvrated['movie_id']==rec.id]['rating']))
        #     reco[rec.id] = rec.vote_average
        recof = dict(sorted(reco.items(), key=lambda item: item[1],reverse=True))
        rmi = list(recof.keys())
        rmid=[]
        for i in mvrated:
            rmid.append(i['movie_id'])
        rmids = [x for x in rmi if x not in rmid]
        #rmids = rmids
        rmvs = Movie4.objects.filter(pk__in=rmids)

        context = {'movie_list': movie_list,'rmvs':rmvs}
        return render(request, 'recommend/recommend.html', context)
    else:
        context ={'text':"Provide some Ratings to enjoy Recommendations"}
        return render(request, 'recommend/recommend.html', context)


# Register user
def signUp(request):
    form = UserForm(request.POST or None)

    if form.is_valid():
        user = form.save(commit=False)
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user.set_password(password)
        user.save()
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("index")

    context = {'form': form}

    return render(request, 'recommend/signUp.html', context)


# Login User
def Login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect("index")
            else:
                return render(request, 'recommend/login.html', {'error_message': 'Your account disable'})
        else:
            return render(request, 'recommend/login.html', {'error_message': 'Invalid Login'})

    return render(request, 'recommend/login.html')


# Logout user
def Logout(request):
    logout(request)
    return redirect("login")
