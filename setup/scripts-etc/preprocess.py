import string, pydub

corpus_dir = '/home/caitlinr/scratch/capt/data/captini/'
audio_dir = corpus_dir + 'audio_correct_names/'
exclude_dir = corpus_dir + 'excluded_audio/'
original_data = corpus_dir + 'metadata_inspect.tsv'
output_dir = '/home/caitlinr/work/pronunciation-score-icelandic/setup/scripts-etc/'

key_file = output_dir + 'KEY.tsv'
data_file = corpus_dir + 'recording_data.tsv'
move_file = output_dir + 'local/move_exclude.sh'

# exclude typo sentences 
# and words missing from the current pronunciation dictionary e.g. írárfoss
excl = ['Er það Björg eða Björg?','Hvar taskan þín? Hún er á borðinu.','Flestir Bandaríkjamenn borða kalkún á þakkargjörðarhátíðinni.', 'Seljalandsfoss, Skógafoss og Írárfoss.']



# 1. Write a key file
def snorm(s):
    return s.lower().translate(str.maketrans('', '', string.punctuation))

recs = [l.split('\t') for l in open(original_data,'r').read().splitlines()]
sents = list(set([l[3] for l in recs[1:]]))
sents = sorted(sents, key = lambda x: snorm(x))
    
o = open(key_file,'w')
o.write('Exercise_ID\tAlignerForm\tOriginal\n')
prev = (0,'')
for s in sents:
    normed = snorm(s)
    if s in excl:
        o.write('EXCL\t'+normed+'\t'+s+'\n')
    else:
        if normed == prev[1]:
            ex = prev[0]
        else:
            ex = prev[0]+1
        o.write('{:0>4}'.format(ex)+'\t'+normed+'\t'+s+'\n')
        prev = (ex,normed)
o.close()



# 2. Write a new version of the metadata file,
# including exercise IDs and normalised text according to the key
kf = [l.split('\t') for l in open(key_file,'r').read().splitlines()[1:]]
kdict = {l[2]:(l[0],l[1]) for l in kf}

o2 = open(data_file,'w')
o2.write('recording_' + '\t'.join(recs[0][:3]))
o2.write('\texercise_id\talign_text\toriginal_text\t')
o2.write('\t'.join(recs[0][4:]) +'\n') 
for r in recs[1:]:
    s = r[3]
    o2.write('\t'.join(r[:3])+'\t')
    o2.write('\t'.join([kdict[s][0],kdict[s][1],s])+'\t')
    o2.write('\t'.join(r[4:]) + '\n')
o2.close()



# check to exclude recordings
# - if the sentence is excluded
# - if the duration is under 0.1 second
# - if there is no speech
# Note: The Empty column from original metadata is too strict
#   as it excludes all recordings under 1.0 seconds
def check(rline,wpath):
    if (rline[3] == 'EXCL') or ('0.0' in rline[13]):
        return False
    else:
        wave = pydub.AudioSegment.from_wav(wpath)
        if pydub.silence.detect_silence(wave):
            return True
        else:
            return False

o3 = open(move_file,'w')
rdf = [l.split('\t') for l in open(data_file,'r').read().splitlines()[1:]]

tt = 0
tot = len(rdf)
for l in rdf:
    wpath = audio_dir+l[1]+'/'+l[2]
    if not check(l,wpath):
        o3.write('mv '+ wpath + ' ' + exclude_dir +'\n')
    tt += 1
    if tt % 250 == 0:
        print('Progress:',tt, '/',tot)
    
o3.close()



