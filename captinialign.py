import shutil, os, subprocess, typing
from random import choice
from collections import defaultdict

class AlignmentError(RuntimeError):
    ...

class AlignOneFunction():

    def __init__(
        self,
        exercise_text,
        rec_file_path,
        rec_duration,
        work_dir='./alignment/new/',
        mfa_dir='./alignment/captini_pretrained_aligner/',
        pdict_name='1_CAPTINI',
    ):
        self.work = work_dir
        self.mfa_dir = mfa_dir
        self.pdict_name = pdict_name
        self.log_path = self.work+'log.txt'
        
# !--- DO NOT CHANGE BELOW HERE 
#	if you are updating the dictionary or acoustic models ---!
        
        self.silence_phone = "sil"
        self.silence_word = "<eps>" # !! TODO
        
        self.rec_path = rec_file_path
        self.normalised_text = exercise_text
        self.duration = rec_duration
        self.long_id = rec_file_path.split('/')[-1].replace('.wav','')
        
        # files already exist:
        self.disambig_path = self.mfa_dir+'dictionary/phones/disambiguation_symbols.int'
        self.word_boundary_int = self.mfa_dir+'dictionary/phones/word_boundary.int'
        self.phone_map_path = self.mfa_dir+'dictionary/phones/phones.txt'
        self.tree_path = self.mfa_dir+'pretrained_aligner/tree'
        self.model_path = self.mfa_dir+'pretrained_aligner/final.mdl'
        self.lexicon_fst_path = self.mfa_dir+'dictionary/'+self.pdict_name+'/L.fst'
        self.lda_mat = self.mfa_dir+'pretrained_aligner/lda.mat'
        
        self.wav_scp = self.work+self.long_id+'_wav.scp' # files don't exist yet:
        self.segments_scp = self.work+self.long_id+'_segments.scp'
        self.text_int = self.work+self.long_id+'_text.int'
        self.feats_scp = self.work+self.long_id+'_feats.scp'
        self.fsts = self.work+self.long_id+'_fsts.ark'
        self.likelihood = self.work+self.long_id+'_likelihoods.scp'
        
        self.utt2spk = self.work+self.long_id+'_utt2spk.scp' #! this does cmvn by utterance,
        self.spk2utt = self.work+self.long_id+'_spk2utt.scp' # not by speaker (yet)
        self.cmvn_ark = self.work+self.long_id+'_cmvn.ark'
        self.cmvn_scp = self.work+self.long_id+'_cmvn.scp'
        
        self.mfcc_options = {'use-energy': False, 'frame-shift': 10, 'frame-length': 25, 'low-freq': 20, 'high-freq': 7800, 'sample-frequency': 16000, 'allow-downsample': True, 'allow-upsample': True, 'snip-edges': False}
        self.feature_options = {'type': 'mfcc', 'use_energy': False, 'frame_shift': 10, 'frame_length': 25, 'snip_edges': False, 'low_frequency': 20, 'high_frequency': 7800, 'sample_frequency': 16000, 'allow_downsample': True, 'allow_upsample': True, 'uses_cmvn': True, 'uses_deltas': True, 'uses_voiced': False, 'uses_splices': True, 'uses_speaker_adaptation': True, 'use_pitch': False, 'min_f0': 50, 'max_f0': 500, 'delta_pitch': 0.005, 'penalty_factor': 0.1, 'splice_left_context': 3, 'splice_right_context': 3}
        self.align_options = {'transition_scale': 1.0, 'acoustic_scale': 0.1, 'self_loop_scale': 0.1, 'beam': 10, 'retry_beam': 40, 'boost_silence': 1.0, 'optional_silence_csl': '1'}

        self.phone_map = {}
        self.setup()
        
    
    def setup(self):
        def temp_file(fpath,contents):
            f = open(fpath,'w')
            f.write(contents)
            f.write('\n')
            f.close()
    
        wav_scp_contents = self.long_id+' ' +self.rec_path
        temp_file(self.wav_scp, wav_scp_contents)
        
        segments_scp_contents = self.long_id + ' ' + self.long_id + ' 0.0 ' + str(round(self.duration,2))
        temp_file(self.segments_scp, segments_scp_contents)
        
        #! TODO change this for cmvn by speaker
        temp_file(self.utt2spk,self.long_id+' '+self.long_id)
        temp_file(self.spk2utt,self.long_id+' '+self.long_id)
        
        word_map_path = self.mfa_dir+'dictionary/'+self.pdict_name+'/words.txt'
        with open(word_map_path,'r') as handle:
            word_map = handle.read().splitlines()
        #word_map = [l.split(' ') for l in word_map]
        word_map = [l.split('\t') for l in word_map] 
        word_map = {l[0]:l[1] for l in word_map}
        
        text_int_contents = self.long_id+' '+' '.join([word_map[word] for word in self.normalised_text.split(' ')])
        temp_file(self.text_int, text_int_contents)
        
        with open(self.phone_map_path,'r') as handle:
            self.phone_map = {l.split('\t')[1]:l.split('\t')[0] for l in handle.read().splitlines()}

    
    def process_intervals(self,intervals):
        if len(intervals) == 0:
            raise AlignmentError("No intervals after aligning.")

        words = self.normalised_text.split(' ')
        word_aligns = []
        phone_aligns = defaultdict(list)
        current_word_begin = None
        w_ix = 0
        p_ix = 0
        
        def r2(t):
            return round(float(t),2)

        for l in intervals:
            start, dur, phone_int = l.split(' ')[2:]
            phone_label = self.phone_map[phone_int]
            start , end = float(start), float(start)+float(dur)
            if phone_label == self.silence_phone:
                continue
            current_word_id = f'{w_ix:03d}__{words[w_ix]}'
            phone, position = phone_label.rsplit("_",1)
            if position in {"B", "S"}:
                current_word_begin = start
            phone_aligns[current_word_id].append((f'{p_ix:03d}__{phone}',r2(start-current_word_begin),r2(end-current_word_begin)))
            if position in {"E", "S"}:
                word_aligns.append((current_word_id, r2(current_word_begin), r2(end)))
                current_word_begin = None
                w_ix += 1
            p_ix +=1
        
        for w,s,e in word_aligns: # !! TODO
            if r2(e-s) != phone_aligns[w][-1][2]:
                raise AlignmentError("Alignment failure.")

        return word_aligns, phone_aligns
        
        
    def align(self): 
    
        def make_safe(value: typing.Any) -> str: # from MFA. TODO
            if isinstance(value, bool):
                return str(value).lower()
            return str(value)
            
        def check_call(proc: subprocess.Popen):
        #! from MFA. TODO
            if proc.returncode is None:
                proc.wait()
            if proc.returncode != 0:
                print('Kaldi Error.')
    
        with open(self.log_path, "w") as log_file:
        
        # compile training graphs:

            graph_proc = subprocess.Popen(
                [
                    shutil.which("compile-train-graphs"),
                    f"--read-disambig-syms={self.disambig_path}",
                    self.tree_path,
                    self.model_path,
                    self.lexicon_fst_path,
                    f"ark:{self.text_int}",
                    f"ark:{self.fsts}",
                ],
                stderr=log_file,
                encoding="utf8",
                env=os.environ,
            )
            graph_proc.communicate()
            
            
        # make mfccs
            mfcc_base_command = [shutil.which("compute-mfcc-feats"), "--verbose=2"]
            raw_ark_path = self.feats_scp.replace(".scp", ".ark")
            for k, v in self.mfcc_options.items():
                mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
            
            mfcc_base_command += ["ark:-", "ark:-"]
            
            seg_proc = subprocess.Popen(
                [
                    shutil.which("extract-segments"),
                    f"scp:{self.wav_scp}",
                    self.segments_scp,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            
            mfcc_proc = subprocess.Popen(
                mfcc_base_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=seg_proc.stdout,
                env=os.environ,
            )  
            
            copy_proc = subprocess.Popen(
                [
                    shutil.which("copy-feats"),
                    "--verbose=2",
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{raw_ark_path},{self.feats_scp}",
                ],
                stdin=mfcc_proc.stdout,
                stderr=log_file,
                env=os.environ,
                encoding="utf8",
            )
            
            for line in mfcc_proc.stderr:
                line = line.strip().decode("utf8")
                log_file.write(line + "\n")
            check_call(copy_proc) # TODO

            subprocess.call(
                [
                    shutil.which("compute-cmvn-stats"),
                    f"--spk2utt=ark:{self.spk2utt}",
                    f"scp:{self.feats_scp}",
                    f"ark,scp:{self.cmvn_ark},{self.cmvn_scp}",
                ],
                stderr=log_file,
                env=os.environ,
            )

        # do the aligning
            feat_string = f"ark,s,cs:apply-cmvn --utt2spk=ark:{self.utt2spk} scp:{self.cmvn_scp} scp:{self.feats_scp} ark:- |"
            feat_string += f" splice-feats --left-context={self.feature_options['splice_left_context']} --right-context={self.feature_options['splice_right_context']} ark:- ark:- |"
            feat_string += f" transform-feats {self.lda_mat} ark:- ark:- |"
            align_proc = subprocess.Popen(
                [
                    shutil.which("gmm-align-compiled"),
                    f"--transition-scale={self.align_options['transition_scale']}",
                    f"--acoustic-scale={self.align_options['acoustic_scale']}",
                    f"--self-loop-scale={self.align_options['self_loop_scale']}",
                    f"--beam={self.align_options['beam']}",
                    f"--retry-beam={self.align_options['retry_beam']}",
                    "--careful=false",
                    self.model_path,
                    f"ark:{self.fsts}",
                    feat_string,
                    "ark:-",
                    f"ark,t:{self.likelihood}",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                encoding="utf8",
                env=os.environ,
            )
            
            lin_proc = subprocess.Popen(
                [
                    shutil.which("linear-to-nbest"),
                    "ark:-",
                    f"ark:{self.text_int}",
                    "",
                    "",
                    "ark:-",
                ],
                stdin=align_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            
            align_words_proc = subprocess.Popen(
                [
                    shutil.which("lattice-align-words"),
                    self.word_boundary_int,
                    self.model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=lin_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            
            phone_proc = subprocess.Popen(
                [
                    shutil.which("lattice-to-phone-lattice"),
                    self.model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stdin=align_words_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            nbest_proc = subprocess.Popen(
                [
                    shutil.which("nbest-to-ctm"),
                    "--print-args=false",
                    f"--frame-shift={round(self.feature_options['frame_shift']/1000,4)}",
                    "ark:-",
                    "-",
                ],
                stdin=phone_proc.stdout,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            
            intervals = []
            log_likelihood = None
            for line in nbest_proc.stdout:
                line = line.strip()
                if not line:
                    continue
                intervals.append(line)
            nbest_proc.wait()

        word_aligns, phone_aligns = self.process_intervals(intervals)
        return word_aligns, phone_aligns
      
        
def makeAlign(exercise_text,user_file_path,rec_duration):
    do_align = AlignOneFunction(exercise_text,user_file_path,rec_duration)
    return do_align.align()
    



