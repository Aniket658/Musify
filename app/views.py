from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.utils.datastructures import MultiValueDictKeyError
from . models import Song
from . import jiosaavn
from . import songtolyrics
from . import predict
from . import facerecog
from . import record
from . import voiceemotion
from . import recommend
# Create your views here.
def home(request):
    return render(request,'home.html')

def playSong(request):
    if request.method=='POST':    
        query=request.POST['query']
    elif request.method=='GET':
        query=request.GET['query']
    lyrics = "false"
    songs=[]
    if query:
        extract = "true"
        songs = jiosaavn.search_for_song(query,lyrics, False)
        print(songs)
        song = Song.objects.filter(songname=songs[0]['song'])
        if (len(song)==0):
            n_song = Song.objects.create(songname=songs[0]['song'], media=songs[0]['media_url'], img=songs[0]['image'])
        else:
            extract = "false"
            rsong = recommend.cosine_sim(song[0].genre, songs[0]['song'])
            getRecom = []
            for s in rsong:
                getsong = Song.objects.get(songname=s) 
                getRecom.append(getsong)
            artist = []
            lsingers = songs[0]['singers'].split(',')[0]
            gartist = jiosaavn.search_for_song(lsingers,lyrics, True)
            for g in gartist:
                artist.append(g)
    else:
        error = {
            "status": False,
            "error":'Query is required to search songs!'
        }
        print(error)
    print(extract)
    return render(request,"playsong.html",{"song":songs[0], "extractFeature":extract, "recom":getRecom, "artist":artist, "aname":lsingers})

def extractFeatures(request):
    song = request.GET['song']
    media = request.GET['media']
    genre = predict.predictGenre(media, song) 
    print(genre)         
    print("Genre Found")
    songtolyrics.getlyrics(song, genre)
    song = Song.objects.filter(songname=song)[0]
    song.genre = genre
    song.save()
    return HttpResponse("Done.")

def mood(request):
    return render(request,"mood.html")

def faceRecog(request):
    emotion = facerecog.face()
    print(emotion)
    return HttpResponse(emotion)

def voiceRecog(request):
    record.recordvoice()
    emotion = voiceemotion.voice()
    print(emotion)
    return HttpResponse(emotion)
