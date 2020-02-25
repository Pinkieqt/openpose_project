import os
import moviepy.editor as mp
#Odkud
directory = '/media/pinkie/A6FC43F0FC43B977/Users/Pinkie/Downloads/uTorrent/Dva a půl chlapa'

#Kam
directoryForSaving = "/media/pinkie/NOVY SVAZEK/dva a pul chlapa/"

counter = 0


for filename in os.listdir(directory):
    counter += 1
    if filename.endswith(".avi"):
        pathToFile = directory + "/" + filename
        clip = mp.VideoFileClip(pathToFile)
        filenameForSaving = filename.replace('.avi', '') + ".mp3"

        clip.audio.write_audiofile(directoryForSaving +  filenameForSaving)
        #print(filename.replace('.avi', '') + ".mp3")
    else:
        print("other file")

print("done")
