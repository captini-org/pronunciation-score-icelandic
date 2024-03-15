#!/bin/bash

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# HELP fonction
Help(){
    echo "usage: ./run.sh [-h] [-o] [-m ACOUSTIC_MODEL_PATH] [-d DICTIONARY_PATH] [-td OUTPUT_DICTIONARY_PATH {ACOUSTIC_MODEL_PATH}]"
    echo ""

    echo "options: "
    echo "  -h, --help                show this help message and exit"
    echo "  -o, --overwrite           overwrite output files when they exist, default is False"
    echo "  -m, --model               use the acoustic model put in parameter to align files"
    echo "  -d, --dictionary          use the dictionary put in parameter instead of using the one created during the run"
    echo "  -td, --pronun_dictionary   create another dictionary adding the pronunciation dictionary in the path mut in parameters. The user can also put an acoustic model in parameters to train the dictionary, instead of using the one created during the run."
}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# Colors for text
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
NC='\033[0m'              # No Color

# Getting the informations contained in info.json
path_to_data="$(grep -o '"path_to_data": "[^"]*' local/info.json | grep -o '[^"]*$')"
output="$(grep -o '"output_folder": "[^"]*' local/info.json | grep -o '[^"]*$')""/"

dict_path=$output"$(grep -o '"dictionary_file": "[^"]*' local/info.json | grep -o '[^"]*$')"
lex_path=$output"$(grep -o '"lexicon_file": "[^"]*' local/info.json | grep -o '[^"]*$')"
model_file="$(grep -o '"MFA_model_name": "[^"]*' local/info.json | grep -o '[^"]*$')"
data_folder="$(grep -o '"data_folder": "[^"]*' local/info.json | grep -o '[^"]*$')"
audio_extension="$(grep -o '"audio_extension": "[^"]*' local/info.json | grep -o '[^"]*$')"
text_extension="$(grep -o '"text_extension": "[^"]*' local/info.json | grep -o '[^"]*$')"
model_path=$PWD"/""$output"$model_file
log="$output""log"
N_audio_file=$(find "$path_to_data"/"$data_folder"/* -name "*."$audio_extension | wc -l)

# If the output folder doesn't exist, it creates one. It will contain all the files the tool creates (lexicon, dictionary, aligned text files)
if [ ! -d "$output" ] 
then
    mkdir $output
fi

# If a log folder doesn't exist, it creates one. This folder will contain the log files that may be created in the different steps
if [ ! -d "$output""log" ] 
then
    mkdir $output'log'
fi

# Initialize the options 
ov=false # Overwrite 
qu=false # Quiet
mo=false # Acoustic model
di=false # Dictionary
pronun_dict=false # Pronunciation dictionary

# We get all the options the users wrote. 
while [ ! -z "$1" ]
do
    case $1 in
        # If the user wants to display the help message.
        -h|--help)
            Help
            exit 0
            ;;
        # If the user wrote '-o' or '--overwrite', the the 'ov' variable will be changed to 'true'
        # This option means the user wants to overwrite on existing files
        -o|--overwrite)
            ov=true
            ;;
        # If the user wrote '-q' or '--quiet', the the 'qu' variable will be changed to 'true'
        # This option means the user wants the tool to display the less things possible 
        -q|--quiet)
            qu=true
            ;;
        # If the user wrote '-m' or '--model', the the 'mo' variable will be changed to 'true'``
        # This option, which should be followed by the path of an acoustic model, will use the model to align files
        -m|--model)
            mo=true
            shift
            if [ -f "$1" ]; then
                model_path=$1
            else 
                printf "${Red}FAILURE${NC} : The model path has not been initialized.\n"
                exit 1
            fi
            ;;
        # If the user wants to use a pre-existing dictionary, he can put this option with the path to the dictionary in parameter.
        -d|--dictionary)
            di=true
            shift
            if [ -f "$1" ]; then
                dict_path=$1
            else 
                printf "${Red}FAILURE${NC} : The dictionary path has not been indicated.\n"
                exit 1
            fi
            ;;
        # This will create another dictionary, whose path has to be in parameter, containing the pronunciations probabilities.
        # The user can put a pre-existing acoustic model in parameter to train the model. By default it will use the model created previously.
	    -td|--pronun_dictionary)
	        pronun_dict=true
            shift
            if [ ! -z "$1" ]; then
                output_dict=$1
            else
                printf "${Red}FAILURE${NC} : The output dictionary name has not been indicated.\n"
            fi

            shift
            if [ -f "$1" ]; then
                acoustic_model=$1
            else
                acoustic_model=$model_path".zip"
	        fi
	        ;;
        # If none of the past option exist, then it displays an error.
	    *)
	        printf "${Red}FAILURE${NC} : The option hasn't been recognized. \n"
	        exit 1 
    esac
shift
done

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# CONTROL PANEL
from_stage=0
to_stage=6

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=0
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
echo '========== Preparing folders =========='
# We call the folder_prep.py program, using the informations in the info.json file and putting the errors messages in prep.log file located in log folder.
# If the user wants to overwrite on the potentially pre-existing .txt files, we add the '-o' option after the python command. It will be recognized by the folder_prep.py program.

if [ $ov == true ]
then
    python3 local/folder_prep.py local/info.json $N_audio_file -o 2> "$log"/prep.log
else 
    python3 local/folder_prep.py local/info.json $N_audio_file 2> "$log"/prep.log
fi

# If an arror occured in the preparation of folder, it show a message and stop the program.
if [ "${?}" -eq 1 ]
then
    # In the end of every log file, we put the date and time of the last modification.
    echo "Last modification : ""$(date)" >> "$log"/prep.log
    printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/prep.log
    exit 1
else
    printf "${Green}SUCCESS${NC} : Preparation of folder.\n "
fi
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=1
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
echo '========== Making lexicon ==========' 

# If the user doesn't want to overwrite on existing file and if there is an existing file, or if the user gave an existing dictionary and therefore does not need to create a lexicon, it displays a message and go to the next step. 
if [ $ov == false ] && [ -f "$lex_path" ] || [ $di == true ] && [ -f "$dict_path" ]; then
    printf "${Yellow}NOTHING DONE${NC} : lexicon or dictionary already existing.\n "
else
    python3 local/make_lexicon.py local/info.json 2> "$log"/lex.log

    if [ "${?}" -ne 0 ] 
    then
        echo "Last modification : ""$(date)" >> "$log"/lex.log
        printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information. \n " "$log"/lex.log
        exit 2
    else
        printf "${Green}SUCCESS${NC} : Lexicon made. \n"
    fi
fi
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=2
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
echo '========== Making dictionnary =========='

if [ $ov == false ] && [ -f "$dict_path" ] || [ $di == true ] && [ -f "$dict_path" ]; then
	printf "${Yellow}NOTHING DONE${NC} : %s already existing.\n " "$dict_path"
else
    # We create a dictionary of phoneme  using the lexicon, the model 'ipd_clean_slt2018.mdl'.
    python3 -m g2p --model ipd_clean_slt2018 --apply "$lex_path" --encoding='utf-8' 1> "$dict_path" 2> "$log"/dict.log

    if [ "${?}" -ne 0 ] 
    then
        echo "Last modification : ""$(date)" >> "$log"/dict.log
        printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/dict.log
        exit 3
    else
        printf "${Green}SUCCESS${NC} : Dictionnary made. \n"
    fi
fi
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=3
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
echo '========== Validating the data =========='

if $qu
then
    mfa validate --quiet "$path_to_data""$data_folder" "$dict_path" 2>&1 "$log"/val.log
else
    mfa validate "$path_to_data""$data_folder" "$dict_path" 2> "$log"/val.log
fi

if [ "${?}" -ne 0 ] 
then
    echo "Last modification : ""$(date)" >> "$log"/val.log
    printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/val.log
    exit 4
else
    printf "${Green}SUCCESS${NC} : Data validated. \n"
fi 
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=4
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
# If the user want to use a pre-existing model, it will then use this model to align the files of the dataset.
if $mo
then
echo '========== Adapting the dataset & align the files using the model =========='
    if $qu
    then
        if $ov 
        then
            mfa align --output_format json -quiet --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$model_path" "$output""out" 2> "$log"/align.log
        else
            mfa align --output_format json --quiet --clean "$path_to_data""$data_folder" "$dict_path" "$model_path" "$output""out" 2> "$log"/align.log
        fi
    else
        if $ov
        then
            mfa align --output_format json --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$model_path" "$output""out" 2> "$log"/align.log
        else
            mfa align --output_format json --clean "$path_to_data""$data_folder" "$dict_path" "$model_path" "$output""out" 2> "$log"/align.log
        fi
    fi

    if [ "${?}" -ne 0 ] 
    then
        echo "Last modification : ""$(date)" >> "$log"/align.log
        printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/align.log
        exit 5
    else
        printf "${Green}SUCCESS${NC} : Alignment finished. \n"
    fi     

# If the user wants to create a new model, it will do so and use this model to align the files of the dataset.
else 

    echo '========== Creating and Training the MFA model =========='

    # If a out folder doesn't exist, it creates one. It will contain every files the 
    if [ ! -d "$output""out" ] 
    then
        mkdir $output'out'
    fi

    if [ $ov == false ] && [ -f "$output""$model_file"".zip" ]
    then
        printf "${Yellow}NOTHING DONE${NC} : %s already created.\n " "$output""$model_file"

    else
        if $qu
        then
            mfa train --output_model_path "$model_path" --quiet --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$output""out" 2>&1 "$log"/train.log
        else
            mfa train --output_model_path "$model_path" --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$output""out" 2> "$log"/train.log
        fi

        if [ "${?}" -eq 1 ] 
        then
            echo "Last modification : ""$(date)" >> "$log"/train.log
            printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/train.log
            exit 5
        else
            printf "${Green}SUCCESS${NC} : Training finished. \n"
        fi        
    fi
fi
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=5
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then

# Ths stage will check if the alignment has been done for all the files. If not, it will put all the non-aligned files in a separate folder.
# It will always display the percentage of missing folders
echo '========== Checking =========='   

N_file=$(expr $(find "$path_to_data"/"$data_folder"/* -name "*."$audio_extension | wc -l) + $(find "$path_to_data"/"$data_folder"/* -name "*."$text_extension | wc -l) )
python3 local/alignment_check.py local/info.json $N_file 2> "$log"/check.log

if [ "${?}" -ne 0 ] 
then
    echo "Last modification : ""$(date)" >> "$log"/check.log
    printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/check.log
    exit 6
else
    printf "${Green}SUCCESS${NC} : Checking finished. \n"
fi    
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

current_stage=6
if [ $pronun_dict == true ]; then
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
echo '========== Training the dictionary =========='
    if $qu
    then
	if $ov
	then
        mfa train_dictionary --quiet --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$acoustic_model" "$output_dict" 2>&1 "$log"/pronun_dict.log
	else
	    mfa train_dictionary --quiet --clean "$path_to_data""$data_folder" "$dict_path" "$acoustic_model" "$output_dict" 2>&1 "$log"/pronun_dict.log
	fi
    else
	if $ov
	then
	    mfa train_dictionary --clean --overwrite "$path_to_data""$data_folder" "$dict_path" "$acoustic_model" "$output_dict" 2> "$log"/pronun_dict.log
	else
	    mfa train_dictionary --clean "$path_to_data""$data_folder" "$dict_path" "$acoustic_model" "$output_dict" 2> "$log"/pronun_dict.log
        fi
    fi
fi

if [ "${?}" -ne 0 ]
then
    echo "Last modification : ""$(date)" >> "$log"/val.log
    printf "${Red}FAILURE${NC} : Something went wrong. See the %s file for more information.\n " "$log"/pronun_dict.log
    exit 4
else
    printf "${Green}SUCCESS${NC} : Dictionary trained. \n"
fi
fi

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

printf "${Green}==================== Program finished without errors ====================${NC}\n "
