# VLC Controller

## Instalacija
Pored requirements-a koji su definisani na [linku](https://github.com/ftn-ai-lab/sc-2019-siit/blob/master/okruzenja/requirements.txt), potrebno je instalirati i *python-vlc* modul.

```bash
pip install python-vlc
```

## Pokretanje aplikacije
U folderu VLController se nalazi sačuvan istreniran model aplikacije. Njegova implementacija se nalazi u fajlu ```tain.py```. Pokretanjem ```auto-player.py``` se pokreće aplikacija. Test primeri se nalaze u fajlu ```test.py```, gde se može videti i procenat tačnosti datih test primera.
##Komande
Dostupne komande u samoj aplikaciji su:


![Mute](vlcontroller/data/mute.jpg) 

Mutiranje zvuka video snimka

![pause](vlcontroller/data/pause.jpg) 

Pauziranje video snimka

![resume](vlcontroller/data/resume.jpg) 

Ponovno pokretanje video snimka

![volumeup](vlcontroller/data/volumeup.jpg) 

Pojačavanje zvuka video snimka

![volumedown](vlcontroller/data/volumedown.jpg) 

Smanjivanje zvuka video snimka

*ESC* dugme omogućava izlazak iz aplikacije.